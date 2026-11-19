"""Measured PETSYS poses, GATE run indexing and batch GUI workflows."""

from types import SimpleNamespace

import numpy as np
import pytest
import uproot

from Gimntools.GimnRec.data_loaders import get_default_parameters
from Gimntools.GimnRec.detector_response import DetectorResponse, ResponseSettings
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.sinogram_io import load_sinogram
from Gimntools.GimnRec.utils import coincidence_to_lor
from test_detector_response import setup_geometry, write_aligned_gimnroot


def test_gate_counts_observed_runs_without_reindexing_the_rotation_plan(tmp_path):
    path = tmp_path / "sparse-runs.root"
    run = np.array([5, 0, 2, 5])
    columns = {f"globalPos{axis}{hit}": np.zeros(4) for axis in "XYZ" for hit in (1, 2)}
    columns.update(crystalID1=np.array([0, 0, 0, 0]), crystalID2=np.array([1, 1, 1, 1]),
                   rsectorID1=np.full(4, 4), rsectorID2=np.full(4, 29), runID=run)
    with uproot.recreate(path) as file:
        file.mktree("Coincidences", columns)
    cfg, angles, sectors = setup_geometry("GATE")
    geometry = {}
    sino, events = coincidence_to_lor(
        path, cfg, angles, get_default_parameters("GATE"), rsectors=sectors,
        geometry_out=geometry, infer_rotation_angles=True)
    assert sino.shape[-1] == 42
    assert sino.sum() == len(events) == 4
    assert geometry["gate_run_ids"] == [0, 2, 5]
    assert geometry["observed_count"] == 3
    assert geometry["count_kind"] == "runs"
    assert geometry["detector_config"]["rotation_angles"] == 6
    np.testing.assert_allclose(geometry["object_angles_deg"], [-150, -60, 0])
    r = siddon_reconstructor(sino, fov_mm=13.44, sino_range_mm=13.44, verbose=False)
    centers = (np.arange(42) + 0.5) * (180 / 42)
    model = DetectorResponse(r, centers, ResponseSettings(0.08184, 3, analytical=True), geometry)
    assert model.doi.nnz > 0


def test_build_archive_and_reconstruct_two_petsys_sinograms(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from Gimntools.GimnRec.ui import app as ui

    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "profiles.db"))
    try:
        app = ui.ReconUI()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        errors = []
        monkeypatch.setattr(app, "error", errors.append)

        def immediate(_label, target, *args):
            target(*args)
            return True

        monkeypatch.setattr(app, "start_worker", immediate)
        paths = tuple(tmp_path / f"sample-{n}.root" for n in (4, 7))
        for path, n in zip(paths, (4, 7)):
            write_aligned_gimnroot(path, poses=n)
        cfg, angles, mapping = setup_geometry("PETSYS")
        cfg.update(sipm_pixel_n=[8, 8], crystal_size=[20, 26.88, 26.88],
                   Half_FOV=35, sino_range_mm=26.88)
        app.petsys.cfg_form._populate(cfg)
        app.petsys.angles.set_data(angles, mapping)
        app.petsys._build_async(paths=paths)
        assert not errors, errors
        assert [entry["name"] for entry in app.sinograms] == ["sample-4_rotang_4", "sample-7_rotang_7"]
        assert [entry["sino"].shape[-1] for entry in app.sinograms] == [60, 105]
        assert app.petsys.cfg_form.get_config()["rotation_angles"] == 6

        app.sinograms[1]["cor_mm"] = 0.14
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        monkeypatch.setattr(ui, "_BatchSinograms",
                            lambda _parent, entries, **_kwargs: SimpleNamespace(result=entries))
        monkeypatch.setattr(ui.filedialog, "askdirectory", lambda **_kwargs: export_dir)
        app.sino_panel._save_files()
        assert not errors, errors
        archived = tuple(export_dir / (entry["name"] + ".npz") for entry in app.sinograms)
        for path, entry in zip(archived, app.sinograms):
            restored = load_sinogram(path)
            np.testing.assert_array_equal(restored["sino"], entry["sino"])
            assert restored["cor_mm"] == entry["cor_mm"]
            assert restored["acquisition"]["measurement"] == path.name.split("_rotang_")[0] + ".root"
        for path in paths:
            path.unlink()
        damaged = tmp_path / "damaged.npz"
        damaged.write_bytes(b"PK\x03\x04")
        app.sinograms.clear()
        monkeypatch.setattr(ui.filedialog, "askopenfilenames",
                            lambda **_kwargs: (archived[0], damaged, archived[1]))
        app.sino_panel._load_files()
        assert len(errors) == 1 and str(errors.pop()).startswith("damaged.npz:")
        assert [entry["acquisition"]["observed_count"] for entry in app.sinograms] == [4, 7]

        panel = app.recon_panel
        panel.v["method"].set("mlem")
        panel.v["iters"].set(1)
        panel.v["gevery"].set(0)
        panel.v["response_mode"].set("Analytical detector response")
        panel._recon_batch()
        assert not errors, errors
        assert [r["name"] for r in app.reconstructions] == [
            "sample-4_rotang_4_mlem", "sample-7_rotang_7_mlem"]
        assert [r["geometry"]["cor_mm"] for r in app.reconstructions] == [0.0, 0.14]
        assert all(np.isfinite(r["volume"]).all() and r["volume"].max() > 0
                   for r in app.reconstructions)
    finally:
        app.destroy()


@pytest.mark.parametrize("mode", ["CrystalID", "GateGlobalPositions", "GeneralGlobalPosition", "PaulGlobalPosition"])
def test_gate_workers_match_with_multiple_geometry_chunks(tmp_path, mode):
    from pandas.testing import assert_frame_equal
    from Gimntools.GimnRec.sinogramer import convert_crystal_to_xy
    from Gimntools.GimnRec.systemSpace import systemSpace

    cfg, detectors, sectors = setup_geometry("GATE")
    cfg["process_as"] = mode
    cfg["rotate_along"] = "z"
    rng = np.random.default_rng(8)
    n = 68_000
    data = {
        "runID": rng.choice([0, 2, 5], n),
        "rsectorID1": np.full(n, 4), "rsectorID2": np.full(n, 29),
        "crystalID1": rng.integers(0, 16, n), "crystalID2": rng.integers(0, 16, n),
        "globalPosX1": rng.normal(30, 0.1, n), "globalPosX2": rng.normal(-31, 0.2, n),
        **{f"globalPos{axis}{hit}": rng.normal(0.1, 3., n) for axis in "YZ" for hit in (1, 2)},
    }
    path = tmp_path / "parallel-gate.root"
    with uproot.recreate(path) as root:
        root.mktree("Coincidences", data)
    arguments = path, cfg, detectors, get_default_parameters("GATE")
    grid = (13, 21, 54)
    single, single_events = coincidence_to_lor(*arguments, rsectors=sectors, matrix=grid, workers=1)
    threaded, threaded_events = coincidence_to_lor(*arguments, rsectors=sectors, matrix=grid, workers=3)
    np.testing.assert_array_equal(single, threaded)
    assert_frame_equal(single_events, threaded_events, check_exact=True)
    assert 0 < threaded.sum() <= n

    oracle = systemSpace(detectors, "z", dict(cfg, crystal_size=np.asarray(cfg["crystal_size"], dtype=float)), "z")
    if mode != "CrystalID":
        oracle.set_matrix(grid)
    expected = []
    for event in (0, 1, 32767, 32768, n - 1):
        pose = 0. if mode == "GeneralGlobalPosition" else -30. * data["runID"][event]
        oracle.rotateBy(pose)
        points = []
        rows = []
        for hit in (1, 2):
            if mode == "CrystalID":
                col, row = convert_crystal_to_xy(data[f"crystalID{hit}"][event], 4, 4)
                rows.append(row)
                local = np.array([-oracle.positionsX[col], -oracle.positionsY[row], 0.])
                point = oracle.getGlobalPreRotation(local, sectors[data[f"rsectorID{hit}"][event]])
            else:
                point = np.array([data[f"globalPos{axis}{hit}"][event] for axis in "XYZ"])
            points.append(oracle.getGlobalRotation(point))
        radial, angle = oracle.global_to_sinogram(*points, "z")
        z = sum(rows) if rows else oracle.get_slice_mono((points[0][2], points[1][2]))
        expected.append((radial, angle, z))
    np.testing.assert_allclose(threaded_events.iloc[[0, 1, 32767, 32768, n - 1]], expected, atol=1e-12)
