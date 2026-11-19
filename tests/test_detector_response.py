"""Synthetic, matched-operator and acquisition checks for DOI and SV-PSF."""

import json

import numpy as np
import pytest
import uproot

from Gimntools.GimnRec.data_loaders import get_default_parameters
from Gimntools.GimnRec.detector_response import (
    DetectorResponse, ResponseSettings, acquisition_geometry,
)
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.utils import coincidence_to_lor


def setup_geometry(fmt):
    config = dict(
        data_from=fmt, process_as="CrystalID", crystal_type="segmented",
        rotate_along="y", rotation_angles=6, detNum=2, arround=180,
        radius=35.0, pla_thickness=2.0, crystal_size=[20, 13.44, 13.44],
        sipm_pitch=[3.36, 3.36], sipm_pixel_n=[4, 4], sino_range_mm=13.44,
        Half_FOV=6.72, slice_thickness_mm=1.68, det_z_rotation=90, e_min=450, e_max=650,
    )
    detector_angles = {"left": 0.2, "right": 180.2}
    mapping = {4: "left", 29: "right"}
    return config, detector_angles, mapping


def lut_file(tmp_path, doi_included=False):
    path = tmp_path / "asymmetric.npz"
    # At x=+5, 80% of the response moves right, with a sub-voxel Y shift.
    kernels = np.array([[[[1.0, 0.0]], [[0.2, 0.8]]]])
    np.savez(path, x_mm=[-5.0, 5.0], y_mm=[0.0],
             dx_mm=[0.0, 1.2], dy_mm=[0.3], kernels=kernels,
             provenance="Synthetic asymmetric calibration", doi_included=doi_included)
    return path


def make_recon(sino=None, **kwargs):
    if sino is None:
        sino = np.zeros((7, 7, 42))
    return siddon_reconstructor(sino, fov_mm=13.44, sino_range_mm=13.44,
                                n_img=12, verbose=False, **kwargs)


def angle_centers(recon):
    return (np.arange(recon.n_ang) + 0.5) * (180.0 / recon.n_ang)


def make_response(recon, mode, tmp_path):
    lut = str(lut_file(tmp_path)) if mode in ("sv", "both") else None
    config, angles, rsectors = setup_geometry("PETSYS")
    geometry = acquisition_geometry("synthetic.root", config, angles, rsectors,
                                    "y", -np.arange(6) * 30)
    doi = mode in ("doi", "both")
    settings = ResponseSettings(0.08 if doi else None, 3.0 if doi else None, lut_path=lut)
    with pytest.warns(UserWarning, match="DOI|SV-PSF"):
        return DetectorResponse(recon, angle_centers(recon), settings, geometry)


@pytest.mark.parametrize("mode", ["doi", "sv", "both"])
def test_full_and_subset_adjointness_and_sensitivity(mode, tmp_path):
    recon = make_recon(dx_cor_mm=0.3, dy_cor_mm=-0.5, cor_mm=0.2)
    response = make_response(recon, mode, tmp_path)
    rng = np.random.default_rng(410)
    image = rng.random((recon.n_img, recon.n_img))
    y = rng.random((recon.n_dist, recon.n_ang))
    project, backproject = response.operators()
    full = project(image)
    assert np.isfinite(full).all() and full.min() >= 0 and full.max() > 0
    assert np.vdot(full, y) == pytest.approx(np.vdot(image, backproject(y)), rel=1e-12)
    pieces = []
    for subset in (np.arange(0, recon.n_ang, 2), np.arange(1, recon.n_ang, 2)):
        f, b = response.operators(subset)
        np.testing.assert_allclose(f(image), full[:, subset], rtol=1e-12)
        assert np.vdot(f(image), y[:, subset]) == pytest.approx(
            np.vdot(image, b(y[:, subset])), rel=1e-12
        )
        pieces.append(b(np.ones_like(y[:, subset])))
    np.testing.assert_allclose(sum(pieces), backproject(np.ones_like(y)), atol=1e-12)


def test_spatially_varying_asymmetric_kernel_mass_and_edge(tmp_path):
    r = make_recon()
    response = make_response(r, "sv", tmp_path)
    n = r.n_img
    y = n // 2
    left, right = y * n + 2, y * n + n - 3
    a = response.sv[:, left].toarray().reshape(n, n)
    b = response.sv[:, right].toarray().reshape(n, n)
    assert a.sum() == pytest.approx(1)
    assert b.sum() == pytest.approx(1)
    assert b[:, n-2:].sum() > a[:, 3:].sum()
    assert response.sv[:, n*n-1].sum() < 1  # no reflecting or edge renormalization


def test_doi_lut_double_count_rejected(tmp_path):
    r = make_recon()
    cfg, det, mapping = setup_geometry("PETSYS")
    geometry = acquisition_geometry("input", cfg, det, mapping, "y", np.arange(6) * 30)
    with pytest.raises(ValueError, match="already includes DOI"):
        DetectorResponse(r, angle_centers(r),
                         ResponseSettings(0.08, 3, lut_path=str(lut_file(tmp_path, True))),
                         geometry)


def test_poisson_updates_with_combined_response(tmp_path):
    recon = make_recon()
    response = make_response(recon, "both", tmp_path)
    forward, _ = response.operators()
    true = np.zeros((recon.n_img, recon.n_img))
    true[5:7, 4:6] = 100
    background = np.full_like(recon.sinogram, 0.1)
    norm = np.linspace(0.8, 1.2, recon.n_dist)[:, None] * np.ones(recon.n_ang)
    recon.sinogram[0] = norm * forward(true) + background[0]

    def likelihood(volume):
        mean = norm * forward(volume[0]) + background[0]
        return (recon.sinogram[0] * np.log(mean) - mean).sum()

    settings = dict(response=response, background_sino=background, norm_sino=norm,
                    regularization_every=0, ratio_cap=None, verbose=False)
    angles = angle_centers(recon)
    early = recon.mlem(angles, iterations=1, **settings)
    later = recon.mlem(angles, iterations=5, **settings)
    assert likelihood(later) > likelihood(early)
    equivalent = recon.osem(angles, iterations=5, subsets_n=1, **settings)
    np.testing.assert_allclose(equivalent, later, atol=1e-10)
    subsets = recon.osem(angles, iterations=3, subsets_n=3, **settings)
    assert np.isfinite(subsets).all() and subsets.min() >= 0 and subsets.max() > 0


def write_root(path, fmt, *, off_lattice=False):
    run, left, right = np.indices((6, 4, 4)).reshape(3, -1)
    chips = (np.full(run.size, 4), np.full(run.size, 29))
    if fmt == "PETSYS":
        x = np.full(run.size, 1.5 * 3.36)
        columns = dict(
            SiPMPosX_1=x, SiPMPosX_2=x,
            SiPMPosY_1=(1.5-left)*3.36 + (0.1 if off_lattice else 0),
            SiPMPosY_2=(1.5-right)*3.36,
            chipID_1=chips[0], chipID_2=chips[1],
            energyCorrected_1=np.full(run.size, 511.0),
            energyCorrected_2=np.full(run.size, 511.0),
            angleGantry_1=run*30.0, slice_1=np.full(run.size, 3),
        )
        tree = "tomographicCoincidences"
    else:
        columns = {f"globalPos{axis}{hit}": np.zeros(run.size)
                   for axis in "XYZ" for hit in (1, 2)}
        columns.update(crystalID1=left, crystalID2=right, runID=run,
                       rsectorID1=chips[0], rsectorID2=chips[1])
        tree = "Coincidences"
    with uproot.recreate(path) as root:
        root.mktree(tree, columns)


@pytest.mark.parametrize("fmt", ["GATE", "PETSYS"])
def test_doi_uses_raw_bins_and_geometry_captured_from_root(fmt, tmp_path):
    path = tmp_path / "measurement.root"
    write_root(path, fmt)
    cfg, detector_angles, mapping = setup_geometry(fmt)
    geometry = {}
    sino, events = coincidence_to_lor(path, cfg, detector_angles, get_default_parameters(fmt),
                                     rsectors=mapping, geometry_out=geometry)
    assert len(events) == 96
    assert sino.sum() == 96
    assert geometry["object_angles_deg"] == list(np.arange(-150.0, 1, 30))
    recon = make_recon(sino)
    angles = angle_centers(recon)
    with pytest.warns(UserWarning, match="nominal crystal-pair"):
        response = DetectorResponse(recon, angles, ResponseSettings(0.08, 3), geometry)
    output = recon.mlem(angles, iterations=1, response=response, verbose=False)
    assert output.max() > 0


def test_petsys_doi_rejects_off_lattice_hits(tmp_path):
    path = tmp_path / "continuous.root"
    write_root(path, "PETSYS", off_lattice=True)
    cfg, detector_angles, mapping = setup_geometry("PETSYS")
    geometry = {}
    sino, _ = coincidence_to_lor(path, cfg, detector_angles, get_default_parameters("PETSYS"),
                                rsectors=mapping, geometry_out=geometry)
    recon = make_recon(sino)
    with pytest.raises(ValueError, match="off-lattice"):
        DetectorResponse(recon, angle_centers(recon), ResponseSettings(0.08, 3), geometry)


@pytest.mark.parametrize("fmt", ["GATE", "PETSYS"])
def test_cli_combined_mode_reports_model_provenance(fmt, tmp_path):
    from importlib import import_module

    main = import_module(f"Gimntools.GimnRec.cli.reconstruct_{fmt.lower()}").main
    cfg, detector_angles, mapping = setup_geometry(fmt)
    for name, value in (("config", cfg), ("angles", detector_angles), ("mapping", mapping),
                        ("calibration", {})):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    root = tmp_path / "measurement.root"
    write_root(root, fmt)
    argv = [
        "--measurement", str(root), "--detector-config", str(tmp_path / "config.json"),
        "--detectors-angles", str(tmp_path / "angles.json"), "--rsectors",
        str(tmp_path / "mapping.json"), "--methods", "osem,mlem", "--mlem-iters", "1",
        "--osem-iters", "1", "--osem-subsets", "2", "--gauss-every", "0",
        "--doi-mu-mm-inv", "0.08", "--doi-face-mm", "3", "--sv-psf-lut",
        str(lut_file(tmp_path)), "--output-dir", str(tmp_path), "--save-format",
        "nrrd", "--no-plots",
    ]
    if fmt == "PETSYS":
        argv += ["--background", str(root), "--background-scale", "0",
                 "--calibration", str(tmp_path / "calibration.json")]
    with pytest.warns(UserWarning, match="SV-PSF"):
        assert main(argv) == 0
    with np.load(tmp_path / "recon_measurement.npz", allow_pickle=False) as out:
        assert out["mlem"].max() > 0 and out["osem"].max() > 0
        model = json.loads(out["geometry_json"].item())["detector_response"]
        assert model["mu_per_mm"] == 0.08
        assert model["lut"]["provenance"] == "Synthetic asymmetric calibration"
        assert model["acquisition"]["measurement"] == str(root)
    import SimpleITK as sitk
    image = sitk.ReadImage(str(tmp_path / "nrrd" / "measurement_mlem.nrrd"))
    assert json.loads(image.GetMetaData("GimnToolsDetectorResponse")) == model


def test_doi_can_model_background_without_signal_support(tmp_path):
    recon = make_recon()
    response = make_response(recon, "doi", tmp_path)
    unsupported = np.flatnonzero(response.doi.getnnz(axis=1) == 0)[0]
    recon.sinogram[0].flat[unsupported] = 5
    background = np.full_like(recon.sinogram, 0.2)
    recon.mlem(angle_centers(recon), iterations=1, response=response,
               background_sino=background, verbose=False)
    with pytest.raises(ValueError, match="no modeled response"):
        recon.mlem(angle_centers(recon), iterations=1, response=response, verbose=False)


def test_doi_symmetry_and_resolution_variation(tmp_path):
    recon = make_recon()
    response = make_response(recon, "doi", tmp_path)
    forward, backward = response.operators()
    sensitivity = backward(np.ones((recon.n_dist, recon.n_ang)))
    np.testing.assert_allclose(sensitivity, sensitivity[::-1, ::-1], atol=1e-12)
    central = np.zeros((recon.n_img, recon.n_img))
    outer = np.zeros_like(central)
    central[5, 5] = 1
    outer[5, 1] = 1
    a, b = forward(central), forward(outer)
    assert not np.isclose(a.sum(), b.sum())
    # An edge source affects a different distribution of raw projection bins.
    assert not np.allclose(a / a.sum(), b / b.sum())


def test_gui_build_then_reconstruct_with_combined_response(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from Gimntools.GimnRec.ui.app import ReconUI

    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "profiles.db"))
    try:
        app = ReconUI()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        errors = []
        app.error = errors.append
        app.start_worker = lambda _name, worker, *args: worker(*args)
        cfg, detector_angles, mapping = setup_geometry("PETSYS")
        root = tmp_path / "measurement.root"
        write_root(root, "PETSYS")
        app.petsys.cfg_form._populate(cfg)
        app.petsys.angles.set_data(detector_angles, mapping)
        app.petsys.var_meas.set(str(root))
        app.petsys._build_async()
        assert not errors, errors
        panel = app.recon_panel
        panel.var_sino.set(app.sinograms[-1]["name"])
        for key, value in dict(method="mlem", iters=1, gevery=0, doi_face="3",
                               doi_mu="0.08", sv_lut=str(lut_file(tmp_path)),
                               response_mode="DOI + residual SV response").items():
            panel.v[key].set(value)
        with pytest.warns(UserWarning, match="SV-PSF"):
            panel._recon_async()
        assert not errors, errors
        result = app.reconstructions[-1]
        model = result["geometry"]["detector_response"]
        assert result["volume"].max() > 0
        assert model["acquisition"]["detector_angles"] == detector_angles
        assert model["acquisition"]["detector_config"]["radius"] == cfg["radius"]
        assert model["lut"]["doi_included"] is False
    finally:
        app.destroy()
