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
