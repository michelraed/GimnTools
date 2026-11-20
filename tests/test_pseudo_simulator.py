import json
from dataclasses import replace
from itertools import combinations
from time import monotonic, sleep

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from Gimntools.GimnRec.listmode import load_listmode, reconstruct_listmode
from Gimntools.GimnRec.listmode_response import prepare_listmode_response
from Gimntools.GimnRec.phantoms import make_phantom
from Gimntools.GimnRec.pseudo_simulator import _interaction, simulate_listmode
from Gimntools.GimnRec.root_listmode import export_root_listmode
from Gimntools.GimnRec.utils import _lor_coordinates, coincidence_to_lor


ANGLES = {"first": 0., "second": -180.}
MAPPING = {0: "first", 15: "second"}


def config(fmt):
    result = dict(data_from=fmt, radius=35., pla_thickness=2., Half_FOV=35.,
                  crystal_size=[20., 26.88, 26.88], sipm_pitch=[3.36, 3.36], sipm_pixel_n=[8, 8],
                  crystal_type="segmented", rotation_angles=12, arround=180., detNum=2, rotate_along="y")
    if fmt == "PETSYS":
        result.update(e_min=450., e_max=650., det_z_rotation=90.)
    else:
        result.update(process_as="CrystalID")
    return result


def test_phantom_packing_and_activity_concentrations():
    derenzo = make_phantom("Derenzo", 24., 20., diameters=(.9, 1.2, 1.5, 1.8, 2.1, 2.4))
    for insert in derenzo.inserts:
        assert np.linalg.norm(insert.center[:2]) + insert.radius <= 12. + 1e-10
    for a, b in combinations(derenzo.inserts, 2):
        distance = np.linalg.norm(np.subtract(a.center, b.center))
        assert distance + 1e-10 >= a.radius + b.radius
    iec = make_phantom("IEC-inspired", 12., 10., contrast=4., diameters=(6.,))
    samples = iec.sample(np.random.default_rng(4), 40000)
    fraction = np.mean(np.linalg.norm(samples, axis=1) < 3.)
    sphere = 4 * np.pi * 3**3 / 3
    body = np.pi * 6**2 * 10
    assert fraction == pytest.approx(4 * sphere / (body + 3 * sphere), abs=.01)
    cold = make_phantom("IEC-inspired", 12., 10., contrast=0., diameters=(6.,))
    assert np.min(np.linalg.norm(cold.sample(np.random.default_rng(4), 1000), axis=1)) >= 3.
    with pytest.raises(ValueError, match="one rod diameter"):
        make_phantom("Derenzo", 24., 20., diameters=(1.,))
    custom = make_phantom("Derenzo", 24., 20., sectors=2, diameters=(1., 2.), sector_angles_deg=(250., 110.))
    assert custom.sector_angles_deg == (250., 110.)
    for a, b in combinations(custom.inserts, 2):
        assert np.linalg.norm(np.subtract(a.center, b.center)) + 1e-10 >= a.radius + b.radius
    with pytest.raises(ValueError, match="total 360"):
        make_phantom("Derenzo", 24., 20., sectors=2, diameters=(1., 2.), sector_angles_deg=(200., 110.))
    with pytest.raises(ValueError, match="extends outside"):
        make_phantom("IEC-inspired", 12., 4., diameters=(6.,))


def test_derenzo_keeps_central_clearance_and_real_intersector_spacing():
    phantom = make_phantom("Derenzo", 24., 16., diameters=(1.4,) * 6,
                           center_clearance_mm=1.5, sector_gap_mm=1.)
    points = np.array([insert.center[:2] for insert in phantom.inserts])
    assert np.min(np.linalg.norm(points, axis=1)) >= 1.5 + .7 - 1e-10
    sectors = np.rint(np.arctan2(points[:, 1], points[:, 0]) / (np.pi / 3)).astype(int) % 6
    assert set(sectors) == set(range(6))
    for i, j in combinations(range(len(points)), 2):
        if sectors[i] != sectors[j]:
            assert np.linalg.norm(points[i] - points[j]) >= 2 * .7 + 1. - 1e-10


def test_iec_background_activity_is_independent_and_can_be_cold():
    rng = np.random.default_rng(14)
    active = make_phantom("IEC-inspired", 12., 10., contrast=4, background_activity=.5, diameters=(6.,))
    samples = active.sample(rng, 40000)
    fraction = np.mean(np.linalg.norm(samples, axis=1) < 3.)
    sphere = 4 * np.pi * 3**3 / 3
    body = np.pi * 6**2 * 10
    assert active.insert_to_background == 8.
    assert fraction == pytest.approx(8 * sphere / (body + 7 * sphere), abs=.01)
    cold = make_phantom("IEC-inspired", 12., 10., contrast=4, background_activity=0, diameters=(6.,))
    assert np.all(np.linalg.norm(cold.sample(rng, 5000), axis=1) <= 3.)
    with pytest.raises(ValueError, match="active background"):
        make_phantom("IEC-inspired", 12., 10., contrast=0., background_activity=0., diameters=(6.,))


def test_detector_interaction_respects_depth_and_pitch_gap():
    lower, upper = np.array([37., -3.36, -3.36]), np.array([57., 3.36, 3.36])
    grid, pitch, pixels = np.eye(2), np.array([3.36, 3.36]), np.array([2, 2])
    beam = np.array([1., 0., 0.])

    def trace(y, travel, centers=False):
        return _interaction(np.array([0., y, 1.7]), beam, lower, upper, grid, pitch, pixels,
                            3., travel, centers, True)

    distance, hit = trace(1.7, 5.)
    assert distance == 42.
    np.testing.assert_allclose(hit, [42., 1.7, 1.7])
    np.testing.assert_allclose(trace(1.7, 5., True)[1], [37., 1.68, 1.68])
    assert np.isinf(trace(0., 0.)[0])
    assert np.isinf(trace(1.7, 21.)[0])


@pytest.mark.parametrize("fmt", ["GATE", "PETSYS"])
def test_acquisition_rotates_source_and_preserves_geometry(tmp_path, fmt):
    phantom = make_phantom("Homogeneous", 2., 2.)
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    settings = dict(primaries=3500, steps=3, start_deg=10., seed=4, chunk_size=1100)
    info = simulate_listmode(a, phantom, config(fmt), ANGLES, MAPPING, **settings)
    simulate_listmode(b, phantom, config(fmt), ANGLES, MAPPING, **settings)
    assert [pose["primaries"] for pose in info["poses"]] == [3500] * 3
    assert [pose["angle_deg"] for pose in info["poses"]] == [10., 70., 130.]
    with np.load(a) as archive, np.load(b) as repeat:
        hits = archive["events"]
        np.testing.assert_array_equal(hits, repeat["events"])
        np.testing.assert_array_equal(np.bincount(archive["step_index"]),
                                      [pose["coincidences"] for pose in info["poses"]])
        assert np.all(hits[:, [3, 7]] == 511.)
        angles = archive["pose_deg"]
        rotate_back = Rotation.from_euler("y", angles[:, None], degrees=True)
        for slice_ in (slice(0, 3), slice(4, 7)):
            scanned = rotate_back.apply(hits[:, slice_][:, [0, 2, 1]])
            np.testing.assert_allclose(np.abs(scanned[:, 0]), 37., atol=1e-9)
            lattice = np.abs(scanned[:, 1:]) / 3.36 - 0.5
            np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-9)
        assert json.loads(archive["metadata_json"].item())["format"] == fmt
    rays, _ = load_listmode(a)
    np.testing.assert_array_equal(rays, hits[:, [0, 1, 2, 4, 5, 6]])


def test_pseudo_crystal_response_recovers_uniform_cylinder_without_phantom_prior(tmp_path):
    path = tmp_path / "uniform.npz"
    cfg = dict(config("PETSYS"), sipm_pixel_n=[4, 4], crystal_size=[20., 13.44, 13.44])
    phantom = make_phantom("Homogeneous", 10., 10.)
    simulate_listmode(path, phantom, cfg, ANGLES, MAPPING, primaries=90000, steps=6, seed=7)
    rays, meta = load_listmode(path, (500., 520.))
    simulator = meta["detector_model"]
    assert len(simulator.event_steps) == len(rays)
    simulator.metadata.pop("phantom")
    shape, spacing = (15, 15, 15), (.84,) * 3
    response = prepare_listmode_response(rays, shape, spacing, (0.,) * 3, simulator)
    assert len(response.operators) == 6
    assert response.event_rows.shape == (len(rays),)
    rng = np.random.default_rng(5)
    a = response.operators[0]
    x, y = rng.random(a.shape[1]), rng.random(a.shape[0])
    np.testing.assert_allclose((a @ x) @ y, x @ (a.T @ y), rtol=1e-12)
    assert np.ptp(response.sensitivity) > 0

    baseline, _ = reconstruct_listmode(rays, shape, spacing, iterations=4, subsets=2)
    corrected, geometry = reconstruct_listmode(rays, shape, spacing, detector_model=response,
                                               iterations=4, subsets=2)
    mlem, stats = reconstruct_listmode(rays, shape, spacing, detector_model=response,
                                      iterations=1, relaxation=1.)
    assert mlem.ravel() @ response.sensitivity == pytest.approx(stats["traced_events"], rel=2e-6)
    coords = (np.arange(15) - 7) * .84
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    inner = (x*x + y*y <= 3.5**2) & (np.abs(z) <= 3.)

    def cv(volume):
        return volume[inner].std(ddof=1) / volume[inner].mean()

    assert geometry["sensitivity"] == "finite crystal-pair response"
    assert cv(corrected) < .55 * cv(baseline)
    constrained, status = reconstruct_listmode(
        rays, shape, spacing, iterations=1, restrict_to_phantom="auto",
        detector_model=replace(response, phantom={"diameter_mm": 10., "height_mm": 10.}),
    )
    assert status["phantom_support"] and status["sensitivity_floor"] == .01
    assert not constrained[0].any() and not constrained[-1].any()
    assert constrained[7, 7, 7] > 0


def test_export_petsys_preserves_aligned_hits_and_matches_binning(tmp_path):
    import uproot

    root, output = tmp_path / "hits.root", tmp_path / "events.npz"
    records = dict(
        SiPMPosX_1=[1.8, 2.4, 1.8], SiPMPosY_1=[-1.5, 2.5, -1.5],
        SiPMPosX_2=[-1.2, -2.3, -1.2], SiPMPosY_2=[2.1, 1.4, 2.1],
        chipID_1=[0, 0, 0], chipID_2=[15, 15, 15], energyCorrected_1=[501., 515., 300.],
        energyCorrected_2=[521., 505., 511.], slice_1=[7, 7, 7], angleGantry_1=[15., 0., 30.],
    )
    with uproot.recreate(root) as data:
        arrays = {key: np.asarray(values) for key, values in records.items()}
        data.mktree("tomographicCoincidences", {key: col.dtype for key, col in arrays.items()}).extend(arrays)
    cfg = config("PETSYS")
    result = export_root_listmode(root, output, cfg, ANGLES, MAPPING, chunk_size=1)
    assert result["input_coincidences"] == 3
    assert result["coincidences"] == 2
    with np.load(output) as saved:
        raw = saved["events"]
        np.testing.assert_array_equal(saved["pose_deg"], [15., 0.])
        np.testing.assert_array_equal(saved["step_index"], [0, 1])
    # Detector local rotation 90°, raw XY -> scanner (37, raw X, -raw Y).
    np.testing.assert_allclose(raw[1], [37., -2.5, 2.4, 515., -37., 1.4, -2.3, 505.], atol=1e-12)
    _, df = coincidence_to_lor(root, cfg, ANGLES, list(records), rsectors=MAPPING, workers=1)
    distance, angle = _lor_coordinates(raw[:, :2], raw[:, 4:6])
    np.testing.assert_allclose(df["distance"], distance)
    np.testing.assert_allclose(df["angle"], angle)


def test_petsys_listmode_decodes_and_projects_an_aligned_full_lattice(tmp_path):
    import uproot

    cfg = dict(config("PETSYS"), radius=37., sipm_pixel_n=[4, 4], crystal_size=[20., 13.44, 13.44])
    uv = (np.indices((4, 4)).reshape(2, -1).T - 1.5) * 3.36
    shape, step = (17, 11, 11), (1.25,) * 3
    axial_centres = (np.arange(shape[0]) - 8) * step[2]
    mean_z = []
    for translation in (0., 3.75):
        path, archive = tmp_path / f"input_{translation}.root", tmp_path / f"output_{translation}.npz"
        raw = {
            "SiPMPosX_1": uv[:, 0] + translation, "SiPMPosY_1": uv[:, 1],
            "SiPMPosX_2": -uv[:, 0] + translation, "SiPMPosY_2": -uv[:, 1],
            "chipID_1": np.zeros(len(uv), dtype=int), "chipID_2": np.full(len(uv), 15),
            "energyCorrected_1": np.full(len(uv), 511.), "energyCorrected_2": np.full(len(uv), 511.),
            "angleGantry_1": np.zeros(len(uv)),
        }
        with uproot.recreate(path) as file:
            file.mktree("tomographicCoincidences", {k: v.dtype for k, v in raw.items()}).extend(raw)
        export_root_listmode(path, archive, cfg, ANGLES, MAPPING, chunk_size=3)
        rays, meta = load_listmode(archive)
        assert load_listmode(archive, use_detector_geometry=False)[1]["detector_model"] is None
        response = prepare_listmode_response(rays, shape, step, (0.,) * 3, meta["detector_model"],
                                             detector_radius_mm=35.)
        assert response.details["encoded_radius_mm"] == 37.
        assert response.details["detector_radius_mm"] == 35.
        assert response.details["positions"] == 1
        np.testing.assert_allclose(list(response.details["readout_offsets_mm"].values()),
                                   [[translation, 0.], [translation, 0.]], atol=1e-8)
        assert response.name == "measured PETSYS crystal-pair response"
        assert response.details["exposure"] == "assumed equal"
        norm = response.sensitivity.reshape(shape).sum(axis=(1, 2))
        mean_z.append(axial_centres @ norm / norm.sum())
    assert mean_z[1] - mean_z[0] == pytest.approx(3.75, abs=.18)


def test_export_petsys_saves_verified_translation_for_sparse_channels(tmp_path):
    import uproot
    from Gimntools.GimnRec.detector_response import _GIMNROOT_PIXELS

    root, archive = tmp_path / "channels.root", tmp_path / "channels.npz"
    data = {"angleGantry_1": np.array([30.]), "energyCorrected_1": np.array([511.]),
            "energyCorrected_2": np.array([511.])}
    offsets = {"first": (4.1, -1.55), "second": (5.3, 1.7)}
    for hit, chip, channel in ((1, 0, 10), (2, 15, 61)):
        i, j = _GIMNROOT_PIXELS[channel]
        offset = offsets[MAPPING[chip]]
        data[f"chipID_{hit}"] = np.array([chip])
        data[f"channelID_{hit}"] = np.array([channel])
        data[f"posI_{hit}"] = np.array([i])
        data[f"posJ_{hit}"] = np.array([j if hit == 2 else (j + 1) % 8])
        data[f"SiPMPosX_{hit}"] = np.array([(j - 3.5) * 3.36 + offset[0]], dtype=np.float32)
        data[f"SiPMPosY_{hit}"] = np.array([(3.5 - i) * 3.36 + offset[1]], dtype=np.float32)
    with uproot.recreate(root) as file:
        file.mktree("tomographicCoincidences", {k: v.dtype for k, v in data.items()}).extend(data)
    export = export_root_listmode(root, archive, config("PETSYS"), ANGLES, MAPPING)
    assert export["detector_config"]["rotation_angles"] == 1
    for name, offset in offsets.items():
        np.testing.assert_allclose(export["petsys_bin_offsets_mm"][name], offset, atol=2e-6)
    rays, meta = load_listmode(archive)
    prepared = prepare_listmode_response(rays, (15, 15, 15), (2.,) * 3, (0.,) * 3, meta["detector_model"])
    assert prepared.details["alignment"] == "verified ROOT channels"
    assert prepared.operators[0].nnz > 0


def test_export_gate_preserves_sparse_run_ids_and_converts_mev(tmp_path):
    import uproot

    root, output = tmp_path / "gate.root", tmp_path / "events.npz"
    data = {
        "runID": np.array([0, 3]), "rsectorID1": np.array([0, 0]), "rsectorID2": np.array([15, 15]),
        "crystalID1": np.array([28, 28]), "crystalID2": np.array([36, 36]),
        "energy1": np.array([.505, .510]), "energy2": np.array([.511, .513]),
        **{f"globalPos{axis}{hit}": np.full(2, value) for hit, coords in enumerate(((40., 4., 2.), (-41., 3., 1.)), 1)
           for axis, value in zip("XYZ", coords)}
    }
    with uproot.recreate(root) as file:
        file.mktree("Coincidences", {key: col.dtype for key, col in data.items()}).extend(data)
    cfg = config("GATE")
    export_root_listmode(root, output, cfg, ANGLES, MAPPING)
    _, df = coincidence_to_lor(root, cfg, ANGLES, list(data), rsectors=MAPPING, workers=1)
    with np.load(output) as saved:
        hits = saved["events"]
        np.testing.assert_array_equal(saved["step_index"], [0, 3])
        np.testing.assert_array_equal(saved["pose_deg"], [0., 45.])
        np.testing.assert_allclose(hits[:, [3, 7]], [[505, 511], [510, 513]])
    distance, angle = _lor_coordinates(hits[:, :2], hits[:, 4:6])
    np.testing.assert_allclose(df["distance"], distance)
    np.testing.assert_allclose(df["angle"], angle)

    rays, meta = load_listmode(output)
    _, model = reconstruct_listmode(rays, (9, 9, 9), (1.5,) * 3, iterations=1,
                                    detector_model=meta["detector_model"])
    assert model["traced_events"] == 2
    assert model["sensitivity"] == "GATE crystal-pair response"

    export_root_listmode(root, output, cfg, ANGLES, MAPPING, gate_global_hits=True)
    rays, meta = load_listmode(output)
    assert meta["detector_model"] is None
    transformed = Rotation.from_euler("y", -45., degrees=True).apply([[40., 4., 2.], [-41., 3., 1.]])
    np.testing.assert_allclose(rays[1], transformed[:, [0, 2, 1]].ravel())
    export_root_listmode(root, output, dict(cfg, process_as="GeneralGlobalPosition"), ANGLES, MAPPING)
    rays, _ = load_listmode(output)
    np.testing.assert_array_equal(rays, [[40., 2., 4., -41., 1., 3.]] * 2)


def test_pseudo_tab_uses_current_montage_and_opens_listmode(tmp_path, monkeypatch):
    import tkinter as tk

    from Gimntools.GimnRec.ui.app import ReconUI

    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "ui.db"))
    try:
        app = ReconUI()
    except tk.TclError:
        pytest.skip("Tk requires a display")
    try:
        panel = app.pseudo_panel
        app.petsys.cfg_form._populate(config("PETSYS"))
        app.petsys.angles.set_data(ANGLES, MAPPING)
        app.petsys.var_meas.set("/data/sample.root")
        panel._use_current_root()
        assert panel.root_path.get() == "/data/sample.root"
        assert app.notebook.tab(panel, "text") == "Pseudo-simulator"
        panel.primaries.set(3000)
        panel.steps.set("2")
        panel.phantom_type.set("Homogeneous")
        panel.preview()
        assert "Homogeneous" in panel.status.get()
        output = tmp_path / "ui.npz"
        monkeypatch.setattr(panel, "_output", lambda _name: str(output))
        panel._simulate()
        deadline = monotonic() + 30.
        while panel.last_output is None and app._active_task and monotonic() < deadline:
            app.update()
            sleep(.02)
        assert panel.last_output == str(output)
        assert panel.status.get().endswith("coincidences")
        panel._use()
        assert app.listmode_panel.path.get() == str(output)
        assert app.recon_workspace.select() == str(app.listmode_panel)
    finally:
        app.destroy()
