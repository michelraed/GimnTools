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
        app.petsys.var_sino_threads.set(2)
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


def test_petsys_parallel_batches_preserve_counts_and_event_order(tmp_path):
    path = tmp_path / "parallel.root"
    repeat = 25_000
    size = repeat * 4
    zeros = np.zeros(size)
    with uproot.recreate(path) as root:
        root.mktree("tomographicCoincidences", dict(
            SiPMPosX_1=zeros, SiPMPosY_1=zeros, SiPMPosX_2=zeros, SiPMPosY_2=zeros,
            chipID_1=np.tile([4, 4, 99, 4], repeat),
            chipID_2=np.full(size, 29),
            energyCorrected_1=np.tile([511, 511, 511, 300], repeat),
            energyCorrected_2=np.full(size, 511),
            angleGantry_1=np.tile([0, 30, 0, 0], repeat),
            slice_1=np.tile([3, 4, 3, 3], repeat),
        ))
    cfg, _, mapping = setup_geometry("PETSYS")
    angles = {"left": 0, "right": 180}
    args = (path, cfg, angles, get_default_parameters("PETSYS"))
    single, single_events = coincidence_to_lor(*args, rsectors=mapping, workers=1)
    parallel, parallel_events = coincidence_to_lor(*args, rsectors=mapping, workers=4)
    np.testing.assert_array_equal(single, parallel)
    np.testing.assert_array_equal(single_events, parallel_events)
    np.testing.assert_array_equal(parallel_events["slice"], np.tile([3, 4], repeat))
    assert parallel[3, 3, 21] == parallel[4, 3, 28] == repeat
    assert parallel.sum() == 2 * repeat


def write_aligned_gimnroot(path, *, invalid=False, poses=6, chip_ids=(4, 29)):
    from Gimntools.GimnRec.detector_response import _GIMNROOT_PIXELS

    run, left, right = np.indices((poses, 64, 64)).reshape(3, -1)
    i1, j1 = _GIMNROOT_PIXELS[left].T
    i2, j2 = _GIMNROOT_PIXELS[right].T
    offsets = np.array([[0.7, -0.4], [-0.35, 0.22]])
    x1 = ((j1 - 3.5) * 3.36 + offsets[0, 0]).astype(np.float32)
    y1 = ((3.5 - i1) * 3.36 + offsets[0, 1]).astype(np.float32)
    x2 = ((j2 - 3.5) * 3.36 + offsets[1, 0]).astype(np.float32)
    y2 = ((3.5 - i2) * 3.36 + offsets[1, 1]).astype(np.float32)
    if invalid:
        i1 = i1.copy()
        i1[0] = (i1[0] + 1) % 8
    with uproot.recreate(path) as root:
        root.mktree("tomographicCoincidences", dict(
            SiPMPosX_1=x1, SiPMPosY_1=y1, SiPMPosX_2=x2, SiPMPosY_2=y2,
            chipID_1=np.full(run.size, chip_ids[0]), chipID_2=np.full(run.size, chip_ids[1]),
            channelID_1=left, channelID_2=right,
            posI_1=i1, posI_2=i2, posJ_1=j2, posJ_2=j2,  # upstream writer bug
            energyCorrected_1=np.full(run.size, 511.0),
            energyCorrected_2=np.full(run.size, 511.0),
            angleGantry_1=run * (180.0 / poses), slice_1=j1 + j2,
        ))
    return offsets


@pytest.mark.parametrize("encoder_rotation", [0, 90])
def test_aligned_gimnroot_keeps_measured_bins_and_matched_doi(tmp_path, encoder_rotation):
    path = tmp_path / "aligned.root"
    offsets = write_aligned_gimnroot(path)
    cfg, angles, mapping = setup_geometry("PETSYS")
    cfg.update(sipm_pixel_n=[8, 8], crystal_size=[20, 26.88, 26.88],
               sino_range_mm=26.88, det_z_rotation=encoder_rotation)
    geometry = {}
    args = (path, cfg, angles, get_default_parameters("PETSYS"))
    sino, events = coincidence_to_lor(*args, rsectors=mapping, geometry_out=geometry)
    conventional, _ = coincidence_to_lor(*args, rsectors=mapping)
    np.testing.assert_array_equal(sino, conventional)
    np.testing.assert_allclose([geometry["petsys_bin_offsets_mm"][head] for head in ("left", "right")],
                               offsets, atol=2e-6)
    assert len(events) == 6 * 64**2
    rec = siddon_reconstructor(sino, fov_mm=26.88, n_img=18, verbose=False)
    theta = angle_centers(rec)
    with pytest.warns(UserWarning, match="nominal crystal-pair"):
        model = DetectorResponse(rec, theta, ResponseSettings(0.08, 3), geometry)
    measured = sino.sum(axis=0) > 0
    covered = model.doi.getnnz(axis=1).reshape(measured.shape) > 0
    assert np.all(covered[measured])
    forward, back = model.operators()
    rng = np.random.default_rng(23)
    x = rng.random((18, 18))
    y = rng.random(measured.shape)
    assert np.vdot(forward(x), y) == pytest.approx(np.vdot(x, back(y)), rel=1e-12)
    assert np.all(back(np.ones_like(y)) >= 0)


def test_gimnroot_alignment_rejects_wrong_trusted_index(tmp_path):
    path = tmp_path / "bad.root"
    write_aligned_gimnroot(path, invalid=True)
    cfg, angles, mapping = setup_geometry("PETSYS")
    cfg.update(sipm_pixel_n=[8, 8], crystal_size=[20, 26.88, 26.88], sino_range_mm=26.88)
    geometry = {}
    sino, _ = coincidence_to_lor(path, cfg, angles, get_default_parameters("PETSYS"),
                                rsectors=mapping, geometry_out=geometry)
    rec = siddon_reconstructor(sino, fov_mm=26.88, n_img=18, verbose=False)
    with pytest.raises(ValueError, match="1 invalid channel/index hit"):
        DetectorResponse(rec, angle_centers(rec), ResponseSettings(0.08, 3), geometry)


def test_analytical_path_includes_preceding_segments_and_side_exit():
    from Gimntools.GimnRec.detector_response import _scintillator_path

    # Along the normal only the struck crystal contributes.
    normal = _scintillator_path(10.0, 0.0, np.ones(1), np.zeros(1), [0.0, -4.0], 3.0)
    np.testing.assert_allclose(normal, [10.0])
    # At 45 degrees the ray crosses the target and its neighboring crystal,
    # with a 1 mm transverse gap between the two.
    cosine = np.sqrt(0.5)
    ray = np.array([cosine])
    oblique = _scintillator_path(5.0, 0.0, ray, ray, [0.0, -4.0], 3.0)
    np.testing.assert_allclose(oblique, [(1.5 + 2.5) / cosine])


def test_analytical_probability_and_saved_lut(tmp_path):
    from Gimntools.GimnRec.analytical_response import (
        analytical_projection_matrix, generate_analytical_lut,
    )

    recon = make_recon()
    angles = angle_centers(recon)
    cfg, det, mapping = setup_geometry("PETSYS")
    geometry = acquisition_geometry("synthetic", cfg, det, mapping, "y", -np.arange(6)*30)
    settings = ResponseSettings(0.08, 3, face_order=4, depth_order=4)
    matrix = analytical_projection_matrix(recon, angles, settings, geometry)
    efficiency = np.asarray(matrix.sum(axis=0)).ravel()
    assert np.all((efficiency > 0) & (efficiency < 1))
    assert np.ptp(efficiency) > 0
    np.testing.assert_allclose(efficiency, efficiency[::-1], atol=1e-12)
    rng = np.random.default_rng(1)
    x, y = rng.random(matrix.shape[1]), rng.random(matrix.shape[0])
    assert np.dot(matrix @ x, y) == pytest.approx(np.dot(x, matrix.T @ y), rel=1e-12)
    path = tmp_path / "analytic.npz"
    exported = generate_analytical_lut(recon, angles, settings, geometry, path,
                                     sample_count=2, half_support=1, fit_tolerance=1)
    assert exported["kernels"].shape == (2, 2, 3, 3)
    assert np.all(exported["kernels"] >= 0)
    np.testing.assert_allclose(exported["kernels"].sum(axis=(2, 3)), 1)
    assert np.all(exported["fit_scale"] > 0)
    assert exported["relative_error"].max() < 1
    with np.load(path, allow_pickle=False) as lut:
        assert lut["doi_included"].item() is True
    with pytest.warns(UserWarning, match="SV-PSF"):
        response = DetectorResponse(recon, angles, ResponseSettings(lut_path=str(path)))
    assert response.sv.nnz > 0
    assert np.any(response.reference_factors == 0)
    for subset in (None, np.arange(0, recon.n_ang, 2)):
        predict, transpose = response.operators(subset)
        image = x.reshape(recon.n_img, recon.n_img)
        sino = y.reshape(recon.n_dist, recon.n_ang)
        if subset is not None:
            sino = sino[:, subset]
        assert np.vdot(predict(image), sino) == pytest.approx(
            np.vdot(image, transpose(sino)), rel=1e-12)
    wrong_grid = make_recon(dx_cor_mm=1)
    with pytest.raises(ValueError, match="must match"):
        DetectorResponse(wrong_grid, angles, ResponseSettings(lut_path=str(path)))
    cfg["radius"] += 3
    different_detector = acquisition_geometry("different", cfg, det, mapping, "y", -np.arange(6)*30)
    with pytest.raises(ValueError, match="original detector"):
        DetectorResponse(recon, angles, ResponseSettings(lut_path=str(path)), different_detector)


def test_aligned_binning_preserves_physical_sensitivity(tmp_path):
    path = tmp_path / "aligned.root"
    write_aligned_gimnroot(path)
    cfg, angles, mapping = setup_geometry("PETSYS")
    cfg.update(sipm_pixel_n=[8, 8], crystal_size=[20, 26.88, 26.88], sino_range_mm=26.88)
    geometry = {}
    sino, _ = coincidence_to_lor(path, cfg, angles, get_default_parameters("PETSYS"),
                                rsectors=mapping, geometry_out=geometry)
    recon = siddon_reconstructor(sino, fov_mm=26.88, n_img=18, verbose=False)
    theta = angle_centers(recon)
    unaligned = {key: value for key, value in geometry.items() if key != "petsys_bin_centers_mm"}
    settings = ResponseSettings(0.08, 3, analytical=True)
    aligned = DetectorResponse(recon, theta, settings, geometry)
    nominal = DetectorResponse(recon, theta, settings, unaligned)
    assert aligned.doi.shape == nominal.doi.shape
    assert (aligned.doi != nominal.doi).nnz > 0
    np.testing.assert_allclose(aligned.doi.sum(axis=0), nominal.doi.sum(axis=0), atol=1e-12)


def test_upstream_channel_indices_are_not_sequential():
    from Gimntools.GimnRec.detector_response import _GIMNROOT_PIXELS

    np.testing.assert_array_equal(_GIMNROOT_PIXELS[[0, 16, 60]], [[1, 7], [6, 6], [0, 0]])
    assert np.unique(_GIMNROOT_PIXELS, axis=0).shape == (64, 2)


def test_analytical_large_fov_keeps_grid_and_clips_to_detector_gap(tmp_path):
    from Gimntools.GimnRec.analytical_response import generate_analytical_lut
    from Gimntools.GimnRec.detector_response import _pair_rays, crystal_response_settings

    pair = (np.array([37.0, 0.0]), np.array([-37.0, 0.0]),
            (np.array([1.0, 0.0]), np.array([-1.0, 0.0])),
            (np.array([0.0, 1.0]), np.array([0.0, -1.0])),
            (np.array([0.0]), np.array([0.0])), 90.0, 1.0)
    rays, weights = _pair_rays(pair, np.zeros(1), np.ones(1), np.array([8.0]),
                               np.ones(1), 0.08184, 3.0)
    np.testing.assert_allclose(rays, [[37, 0, -37, 0]])
    assert weights[0] > 0

    cfg, det, mapping = setup_geometry("PETSYS")
    cfg.update(sipm_pixel_n=[8, 8], crystal_size=[20, 26.88, 26.88], rotation_angles=12)
    geometry = acquisition_geometry("synthetic", cfg, det, mapping, "y", -np.arange(12)*15)
    r = siddon_reconstructor(np.zeros((1, 15, 180)), fov_mm=70, sino_range_mm=26.88,
                             verbose=False)
    angles = angle_centers(r)
    signals = []
    for material in ("LYSO", "BGO"):
        settings = crystal_response_settings({}, material=material, mu="", face="", analytical=True)
        response = DetectorResponse(r, angles, settings, geometry)
        forward, back = response.operators(np.arange(0, r.n_ang, 2))
        image = np.ones((r.n_img, r.n_img))
        projection = forward(image)
        np.testing.assert_allclose(projection.sum(), back(np.ones_like(projection)).sum())
        signals.append(projection.sum())
        fitting = generate_analytical_lut(r, angles, settings, geometry,
                                          tmp_path / f"{material}.npz", sample_count=3)
        np.testing.assert_allclose(fitting["kernels"].sum(axis=(-2, -1)), 1)
        assert fitting["relative_error"].max() < 0.3
        assert fitting["reference_grid"][1] == 70
    assert signals[0] != pytest.approx(signals[1])

    # A LUT fitted to the default 15×15 image is still usable at 0.84 mm.
    with pytest.warns(UserWarning, match="finer than the radial sampling"):
        fine = siddon_reconstructor(
            r.sinogram, n_img=70, voxel_mm=0.84, sino_range_mm=r.sino_range, verbose=False)
    path = tmp_path / "LYSO.npz"
    lut_settings = ResponseSettings(lut_path=str(path))
    with pytest.raises(ValueError, match="Build the sinogram from ROOT"):
        DetectorResponse(fine, angles, lut_settings)
    with pytest.warns(UserWarning, match="was recomputed"):
        response = DetectorResponse(fine, angles, lut_settings, geometry)
    assert response.metadata["lut"]["reference_factors_recomputed"]
    with np.load(path, allow_pickle=False) as original:
        assert not np.allclose(response.reference_factors, original["reference_factors"])
    for subset in (None, np.arange(0, fine.n_ang, 4)):
        forward, back = response.operators(subset)
        rng = np.random.default_rng(31)
        image = rng.random((70, 70))
        predicted = forward(image)
        y = rng.random(predicted.shape)
        np.testing.assert_allclose(np.vdot(predicted, y), np.vdot(image, back(y)), rtol=1e-12)
    fine.sinogram[0] = response.operators()[0](image)
    reconstructed = fine.osem(
        angles, iterations=3, subsets_n=4, response=response, verbose=False)
    assert reconstructed.shape == (1, 70, 70)
    assert np.isfinite(reconstructed).all() and reconstructed.max() > 0


def test_crystal_presets_use_profile_or_explicit_override():
    from Gimntools.GimnRec.detector_response import crystal_response_settings

    profile = {"crystal_material": "BGO", "crystal_face_width_mm": 2.9, "mu_511_mm_inv": 0.1}
    settings = crystal_response_settings(profile)
    assert (settings.material, settings.face_width_mm, settings.mu_per_mm) == ("BGO", 2.9, 0.1)
    assert crystal_response_settings(profile, material="LYSO").mu_per_mm == pytest.approx(0.08184)
    assert crystal_response_settings(profile, mu="0,09").mu_per_mm == pytest.approx(0.09)
    with pytest.raises(ValueError, match="attenuation coefficient"):
        crystal_response_settings({"crystal_material": "Custom"})


def test_face_order_auto_adapts_to_voxel_and_uses_the_same_adjoint():
    from Gimntools.GimnRec.detector_response import crystal_response_settings, face_quadrature_order

    cfg, angles, mapping = setup_geometry("PETSYS")
    geometry = acquisition_geometry("synthetic", cfg, angles, mapping, "y", -np.arange(6)*30)
    settings = crystal_response_settings({}, face_order="Auto", analytical=True)
    coarse = make_recon()
    assert face_quadrature_order(coarse, settings) < 5
    with pytest.warns(UserWarning, match="finer than the radial sampling"):
        fine = siddon_reconstructor(np.zeros((1, 7, 42)), n_img=16, voxel_mm=0.84,
                                    sino_range_mm=13.44, verbose=False)
    model = DetectorResponse(fine, angle_centers(fine), settings, geometry)
    assert model.metadata["face_order_used"] == 5
    fixed = ResponseSettings(settings.mu_per_mm, settings.face_width_mm, face_order=5, analytical=True)
    reference = DetectorResponse(fine, angle_centers(fine), fixed, geometry)
    assert (model.doi != reference.doi).nnz == 0
    forward, transpose = model.operators(np.arange(0, fine.n_ang, 2))
    x = np.linspace(0, 1, 16**2).reshape(16, 16)
    y = np.arange(fine.n_dist * 21).reshape(fine.n_dist, 21)
    assert np.vdot(forward(x), y) == pytest.approx(np.vdot(x, transpose(y)), rel=1e-12)


def test_gate_lut_saves_a_broad_grid_with_uncaptured_voxels(tmp_path):
    from Gimntools.GimnRec.analytical_response import analytical_projection_matrix, generate_analytical_lut

    cfg, det, mapping = setup_geometry("GATE")
    cfg.update(rotation_angles=12, sipm_pixel_n=[2, 2], crystal_size=[20, 6.72, 6.72], sino_range_mm=6.72)
    acq = acquisition_geometry("gate-two-segment-test", cfg, det, mapping, "y", -15. * np.arange(12))
    r = siddon_reconstructor(np.zeros((1, 3, 36)), n_img=75, voxel_mm=.84, sino_range_mm=6.72,
                             verbose=False)
    angles = angle_centers(r)
    settings = ResponseSettings(.08184, 3.0, face_order="auto", analytical=True)
    matrix = analytical_projection_matrix(r, angles, settings, acq)
    mask = np.asarray(matrix.getnnz(axis=0)).reshape(75, 75) > 0
    assert not mask[2, 2] and mask[37, 37]  # The previous mesh included a nonobservable corner.
    path = tmp_path / "gate_psf.npz"
    with pytest.warns(UserWarning, match="nearby observable"):
        fitted = generate_analytical_lut(r, angles, settings, acq, path, fit_tolerance=1.)
    assert path.is_file()
    np.testing.assert_array_equal(fitted["source_support"], mask)
    assert (fitted["efficiency"] == 0).any()
    assert np.all(fitted["fit_efficiency"] > 0)
    assert fitted["reference_grid"][1] == pytest.approx(63.)
    assert np.allclose(fitted["kernels"].sum(axis=(-1, -2)), 1.)
    with pytest.warns(UserWarning, match="SV-PSF"):
        response = DetectorResponse(r, angles, ResponseSettings(lut_path=str(path)), acq)
    forward, adjoint = response.operators()
    assert not np.any(forward(~mask))
    assert not np.any(adjoint(np.ones((r.n_dist, r.n_ang)))[~mask])
    rng = np.random.default_rng(1909)
    image = rng.random(mask.shape)
    measured = rng.random((r.n_dist, r.n_ang))
    np.testing.assert_allclose(np.vdot(forward(image), measured), np.vdot(image, adjoint(measured)), rtol=1e-12)


def test_depth_auto_matches_resolved_order_and_improves_gated_reference():
    from Gimntools.GimnRec.analytical_response import analytical_projection_matrix
    from Gimntools.GimnRec.detector_response import crystal_response_settings, depth_quadrature_order

    cfg, det, mapping = setup_geometry("GATE")
    cfg.update(sipm_pixel_n=[8, 8], crystal_size=[20, 26.88, 26.88], sino_range_mm=26.88)
    acq = acquisition_geometry("gate-depth-test", cfg, det, mapping, "y", -30. * np.arange(6))
    r = siddon_reconstructor(np.zeros((1, 15, 90)), n_img=23, voxel_mm=1.25, sino_range_mm=26.88,
                             verbose=False)
    angles = angle_centers(r)
    auto = crystal_response_settings(cfg, depth_order="auto", face_order=3, analytical=True)
    order = depth_quadrature_order(r, auto, acq)
    assert 3 < order < 12
    fitted = DetectorResponse(r, angles, auto, acq)
    assert fitted.metadata["depth_order_used"] == order
    fixed = analytical_projection_matrix(r, angles,
                                         ResponseSettings(.08184, 3., 3, order, analytical=True), acq)
    assert (fitted.doi != fixed).nnz == 0
    coarse = analytical_projection_matrix(r, angles, ResponseSettings(.08184, 3., 3, 3), acq)
    reference = analytical_projection_matrix(r, angles, ResponseSettings(.08184, 3., 3, 12), acq)
    assert np.linalg.norm((fitted.doi - reference).data) < np.linalg.norm((coarse - reference).data)


@pytest.mark.parametrize("use_lut", [True, False])
def test_automatic_psf_reconstructs_and_saves_the_fitted_map(tmp_path, monkeypatch, use_lut):
    tk = pytest.importorskip("tkinter")
    from Gimntools.GimnRec import analytical_response
    from Gimntools.GimnRec.ui import app as ui

    monkeypatch.setattr(analytical_response, "DEFAULT_LUT_FIT_TOLERANCE", 1.1 if use_lut else 0.)
    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "profiles.db"))
    try:
        app = ui.ReconUI()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        errors = []
        monkeypatch.setattr(app, "error", errors.append)
        cfg, det, mapping = setup_geometry("GATE")
        acq = acquisition_geometry("gate-automatic", cfg, det, mapping, "y", -30. * np.arange(6))
        r = siddon_reconstructor(np.zeros((1, 7, 42)), fov_mm=13.44, sino_range_mm=13.44, verbose=False)
        angles = angle_centers(r)
        reference = DetectorResponse(r, angles, ResponseSettings(.08184, 3., analytical=True), acq)
        image = np.zeros((7, 7))
        image[3, 3] = 1000.
        counts = reference.operators()[0](image)
        app.set_sinogram(counts[None], r.fov_mm, r.sino_range, "GATE", detector_config=cfg,
                         acquisition=acq, angles_override=angles, name="Automatic")
        p = app.recon_panel
        p.refresh_sino_list(select="Automatic")
        p.v["method"].set("mlem")
        p.v["iters"].set(2)
        p.v["response_mode"].set("Automatic SV-PSF")
        p.v["sv_lut"].set("")
        warning = "SV-PSF" if use_lut else "fit exceeds the error threshold"
        with pytest.warns(UserWarning, match=warning):
            app.run_recon(p._snapshot(), "Automatic")
        assert not errors, errors
        result = app.reconstructions[-1]
        settings = result["geometry"]["detector_response"]
        assert settings["automatic_psf"]["used_lut"] == use_lut
        if use_lut:
            assert settings["lut"]["source"] == "automatic"
        else:
            assert settings["analytical"]
        assert np.isfinite(result["volume"]).all() and result["volume"].max() > 0
        assert not list(tmp_path.glob("*.npz"))
        path = tmp_path / "generated_psf.npz"
        monkeypatch.setattr(ui.filedialog, "asksaveasfilename", lambda **_: str(path))
        p.on_new_volume()
        assert "disabled" not in p.save_psf.state()
        p.save_psf.invoke()
        assert path.read_bytes() == result["psf_lut_bytes"]
        with np.load(path, allow_pickle=False) as data:
            assert data["source_efficiency"].shape == (7, 7)
        if not use_lut:
            p.v["response_mode"].set("Analytical detector response")
            app.run_recon(p._snapshot(), "Automatic")
            np.testing.assert_allclose(result["volume"], app.reconstructions[-1]["volume"])
    finally:
        app.destroy()
