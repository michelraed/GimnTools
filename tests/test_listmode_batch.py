"""ROOT angle provenance and end-to-end batch list mode."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import uproot

from Gimntools.GimnRec.root_listmode import export_root_listmode


HEADS = {"Left": 0., "Right": 180.}
MAPPING = {0: "Left", 1: "Right"}


def _config(fmt):
    cfg = dict(data_from=fmt, radius=30., pla_thickness=2., Half_FOV=30.,
               crystal_size=[15., 6.4, 6.4], sipm_pitch=[3.2, 3.2], sipm_pixel_n=[2, 2],
               crystal_type="segmented", rotation_angles=12, arround=180., detNum=2, rotate_along="y")
    if fmt == "PETSYS":
        cfg.update(e_min=450., e_max=650., det_z_rotation=90.)
    else:
        cfg["process_as"] = "GateGlobalPositions"
    return cfg


def _root(path, tree, data):
    with uproot.recreate(path) as file:
        file.mktree(tree, {key: values.dtype for key, values in data.items()}).extend(data)


@pytest.mark.parametrize("process", ["GateGlobalPositions", "GeneralGlobalPosition", "CrystalID"])
def test_gate_listmode_prefers_recorded_angle_with_units_and_sparse_runs(tmp_path, process):
    root, output = tmp_path / "acquisition.root", tmp_path / "result.npz"
    angles = [13., 31., 61.]
    data = {"runID": np.array([2, 2, 9]), "crystalID1": np.array([0, 0, 0]),
            "crystalID2": np.array([3, 3, 3]), "rsectorID1": np.zeros(3, dtype=int),
            "rsectorID2": np.ones(3, dtype=int), "energy1": np.full(3, .511),
            "energy2": np.full(3, .511), "scan_angle": np.deg2rad(angles)}
    hits = np.array([[33., 1., 2.], [-33., -1., -2.]])
    data.update({f"globalPos{axis}{hit}": np.full(3, value)
                 for hit, vertex in enumerate(hits, 1) for axis, value in zip("XYZ", vertex)})
    _root(root, "Coincidences", data)
    opts = dict(_config("GATE"), process_as=process)
    meta = export_root_listmode(root, output, opts, HEADS, MAPPING,
                               gate_angle_branch="scan_angle", gate_angle_unit="rad", chunk_size=1)
    assert [p["run_id"] for p in meta["poses"]] == [2, 2, 9]
    assert meta["pose_source"] == {"branch": "scan_angle", "unit": "rad"}
    with np.load(output) as archive:
        np.testing.assert_allclose(archive["pose_deg"], angles)
        events = archive["events"][:, [0, 1, 2, 4, 5, 6]].reshape(-1, 2, 3)
    if process == "CrystalID":
        # Sampled hit X is the configured detector face, after undoing
        # the per-event rotation saved in the ROOT.
        for pair, angle in zip(events, angles):
            scanner = Rotation.from_euler("y", angle, degrees=True).apply(pair[:, [0, 2, 1]])
            np.testing.assert_allclose(scanner[:, 0], [32., -32.])
    else:
        for pair, angle in zip(events, angles):
            expected = (hits if process == "GeneralGlobalPosition"
                        else Rotation.from_euler("y", -angle, degrees=True).apply(hits))
            np.testing.assert_allclose(pair, expected[:, [0, 2, 1]])


def test_batch_root_export_and_reconstruction_keep_each_files_angles(tmp_path, monkeypatch):
    import tkinter as tk
    from Gimntools.GimnRec.ui.app import ReconUI

    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "desktop.db"))
    try:
        app = ReconUI()
    except tk.TclError:
        pytest.skip("Tk requires a display")
    try:
        app.withdraw()
        monkeypatch.setattr(app, "start_worker", lambda operation, target, *args: (target(*args), True)[1])
        errors = []
        monkeypatch.setattr(app, "error", errors.append)
        app.petsys.cfg_form._populate(_config("PETSYS"))
        app.petsys.angles.set_data(HEADS, MAPPING)
        paths = []
        pose_lists = ([5., 25.], [11., 29., 58.])
        for i, poses in enumerate(pose_lists):
            folder = tmp_path / str(i)
            folder.mkdir()
            root = folder / "sample.root"
            paths.append(str(root))
            pixel = np.array([[-1.6, -1.6], [-1.6, 1.6], [1.6, -1.6], [1.6, 1.6]])
            a, b = np.tile(pixel, (len(poses), 1)), np.tile(-pixel, (len(poses), 1))
            _root(root, "tomographicCoincidences", {
                "SiPMPosX_1": a[:, 0], "SiPMPosY_1": a[:, 1], "chipID_1": np.zeros(len(a), dtype=int),
                "SiPMPosX_2": b[:, 0], "SiPMPosY_2": b[:, 1], "chipID_2": np.ones(len(a), dtype=int),
                "energyCorrected_1": np.full(len(a), 511.), "energyCorrected_2": np.full(len(a), 511.),
                "angleGantry_1": np.repeat(poses, len(pixel)),
            })
        broken = tmp_path / "broken.root"
        broken.write_text("invalid file")
        paths.insert(1, str(broken))
        occupied = tmp_path / "sample_listmode.npz"
        occupied.write_bytes(b"pre-existing")
        monkeypatch.setattr("tkinter.filedialog.askopenfilenames", lambda **kw: tuple(paths))
        monkeypatch.setattr("tkinter.filedialog.askdirectory", lambda **kw: str(tmp_path))
        app.pseudo_panel._export_batch()
        app._drain()
        assert app.pseudo_panel.status.get() == "2/3 list-mode NPZ exported"
        assert occupied.read_bytes() == b"pre-existing"
        app.pseudo_panel._use()
        panel = app.listmode_panel
        assert len(panel.batch_paths) == 2
        for path, poses in zip(panel.batch_paths, pose_lists):
            with np.load(path) as archive:
                np.testing.assert_allclose(np.unique(archive["pose_deg"]), poses)
        panel.matrix.set("11 11 11")
        panel.voxel.set("1 1 1")
        panel.iterations.set(1)
        panel._start_batch()
        app._drain()
        assert not errors
        assert panel.result.get() == "2/2 list-mode acquisitions reconstructed"
        assert [v["geometry"]["detector_model"]["positions"] for v in app.reconstructions] == [2, 3]
        saved_paths = panel.batch_paths
        app.session.flush()
    finally:
        app.destroy()
    restored = ReconUI()
    try:
        restored.withdraw()
        assert restored.listmode_panel.batch_paths == saved_paths
        assert set(restored.listmode_panel.jobs.set(str(path), "state") for path in saved_paths) == {"Ready"}
    finally:
        restored.destroy()
