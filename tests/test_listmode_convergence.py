"""Coincidence geometry, robust convergence and axial units regressions."""

import numpy as np
import pytest

from Gimntools.GimnRec.convergence import convergence_series, detect_source_regions
from Gimntools.GimnRec.corrections import estimate_sinogram_cor
from Gimntools.GimnRec.listmode import EVENT_COLUMNS, _ray_matrix, load_listmode, reconstruct_listmode
from Gimntools.GimnRec.reconstructor import siddon_reconstructor


def test_npz_columns_energy_and_gantry_frame(tmp_path):
    name = tmp_path / "columns.npz"
    events = np.array([[-9., 0, -2., 510, 9., 0, 2., 512],
                       [-8., 1, 1., 510, 8., 2, -3., 300],
                       [2., 1, 1., 510, 2., 1, 1., 520]])
    np.savez(name, **dict(zip(EVENT_COLUMNS, events.T)), gantry_deg=[90., 0., 0.], rotation_axis="z")
    rays, meta = load_listmode(name, (450, 650))
    np.testing.assert_allclose(rays, [[0, 9, -2, 0, -9, 2]], atol=1e-12)
    assert meta["total_events"] == 3 and meta["accepted_events"] == 1
    with pytest.raises(ValueError, match="energy window"):
        load_listmode(name, (650, 450))


def test_3d_lengths_reverse_adjoint_and_reconstruction(tmp_path):
    shape = np.array([9, 7, 5])
    spacing = np.array([1., 1.5, 2.])
    center = np.array([0.1, -0.2, 0.3])
    point = np.array([1.1, -0.2, 2.3])
    rng = np.random.default_rng(14)
    direction = rng.normal(size=(120, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    rays = np.column_stack((point - 20 * direction, point + 20 * direction))
    matrix = _ray_matrix(rays, shape, spacing, center - shape * spacing / 2)
    reverse = _ray_matrix(np.column_stack((rays[:, 3:], rays[:, :3])),
                          shape, spacing, center - shape * spacing / 2)
    np.testing.assert_allclose(matrix.toarray(), reverse.toarray(), atol=1e-12)
    x = rng.random(shape.prod())
    y = rng.random(len(rays))
    np.testing.assert_allclose((matrix @ x) @ y, x @ (matrix.T @ y))
    cardinal = _ray_matrix(np.array([[-20, 0, 0, 20, 0, 0], [-20, 20, 0, 20, 20, 0]]),
                           shape, spacing, center - shape * spacing / 2)
    np.testing.assert_allclose(np.asarray(cardinal.sum(axis=1)).ravel(), [9, 0])

    data = np.column_stack((rays[:, :3], np.full(len(rays), 510),
                            rays[:, 3:], np.full(len(rays), 520)))
    name = tmp_path / "matrix.npz"
    np.savez(name, events=data)
    coincidences, _ = load_listmode(name)
    kwargs = dict(center=center, iterations=6, subsets=2)
    cached, geom = reconstruct_listmode(coincidences, shape[::-1], spacing, **kwargs)
    streamed, _ = reconstruct_listmode(coincidences, shape[::-1], spacing, **kwargs, cache_bytes=0)
    np.testing.assert_allclose(streamed, cached)
    assert np.unravel_index(cached.argmax(), cached.shape) == (3, 3, 5)
    assert geom["traced_events"] == len(rays)
    normal, geom = reconstruct_listmode(coincidences, shape[::-1], spacing, center=center,
                                        iterations=1, relaxation=1, normalization_lors=rays)
    sensitivity = np.asarray(matrix.sum(axis=0)).ravel()
    np.testing.assert_allclose(normal.ravel() @ sensitivity, len(rays))
    assert geom["sensitivity"] == "independent normalization LORs"


def test_robust_cor_separates_off_axis_source_from_radial_offset():
    angles = (np.arange(180) + 0.5)
    bins = (np.arange(200) - 99.5) * .2
    rad = np.deg2rad(angles)
    centres = 0.7 + 5.4 * np.cos(rad) - 2.3 * np.sin(rad)
    sino = 1000 * np.exp(-((bins[:, None] - centres) / .8)**2)
    sino[:, ::7] = 0
    sino[:, 11] = np.roll(sino[:, 11], 45)
    value, details = estimate_sinogram_cor(sino[None], 40., angles)
    assert value == pytest.approx(.7, abs=.02)
    assert details["used_angles"] < len(angles)
    assert details["position_amplitude_mm"] == pytest.approx(np.hypot(5.4, 2.3), abs=.025)


def test_regions_and_checkpoints_reject_hot_outlier_and_report_contrast():
    z, y, x = np.ogrid[:13, :31, :31]
    phantom = np.ones((13, 31, 31))
    for cy, cx in [(8, 9), (20, 21)]:
        phantom += 15 * np.exp(-((z - 6)**2 + (y - cy)**2 + (x - cx)**2) / 3)
    phantom[5, 12, 24] = 1e6
    regions = detect_source_regions(phantom, (1., 1., 1.5), radius_mm=2)
    assert {r.center_zyx for r in regions} == {(6, 8, 9), (6, 20, 21)}

    calls = []

    def run(iterations, subsets, checkpoint):
        calls.append((iterations, subsets))
        for i in range(1, iterations + 1):
            for k, image in enumerate(phantom):
                checkpoint(k, i, 1 + (image - 1) * i)

    rows = convergence_series(run, regions, [1, 3], [1, 4], cor_mm=.2, label="Trial")
    assert calls == [(3, 1), (3, 4)]
    assert len(rows) == 8
    assert rows[-1]["updates"] == 12
    assert rows[-1]["contrast_pct"] > rows[0]["contrast_pct"] > 0
    assert 1 <= rows[-1]["background"] < 1.25


def test_sensitivity_floor_keeps_zero_support_after_smoothing():
    angles = (np.arange(30) + .5) * 6
    recon = siddon_reconstructor(np.ones((1, 9, 30)), sino_range_mm=9, fov_mm=36,
                                 n_img=36, verbose=False)
    updates = []
    volume = recon.mlem(angles, iterations=2, regularization_every=1, regularization_sigma=.5,
                        verbose=False, sensitivity_floor=.15,
                        checkpoint=lambda z, it, x: updates.append((z, it, x.copy())))
    assert [item[1] for item in updates] == [1, 2]
    assert not volume[0, 0, 0] and volume[0, 18, 18] > 0
    np.testing.assert_array_equal(updates[-1][2], volume[0])


def test_reconstruction_workspace_listmode_and_convergence(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from Gimntools.GimnRec.ui.app import ReconUI

    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "app.db"))
    try:
        app = ReconUI(themed=True)
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    def immediate(_op, target, *args):
        target(*args)
        return True

    try:
        errors = []
        monkeypatch.setattr(app, "start_worker", immediate)
        monkeypatch.setattr(app, "error", errors.append)
        monkeypatch.setattr("Gimntools.GimnRec.ui.app.messagebox.showerror",
                            lambda *_args: pytest.fail("unexpected UI error"))
        angles = (np.arange(30) + .5) * 6
        ref = siddon_reconstructor(np.zeros((5, 21, 30)), sino_range_mm=21, fov_mm=21, verbose=False)
        spot = np.zeros((21, 21))
        spot[9:12, 9:12] = 20
        noise = .2 + ref._project(spot, np.deg2rad(angles))
        volume = np.stack([noise * w for w in (0.10, .50, 1., .50, .10)])
        app.set_sinogram(volume, 21., 21., "PETSYS", detector_config={"slice_thickness_mm": 1.68},
                         angles_override=angles, name="Point")
        app._drain()
        p = app.convergence_panel
        p.iterations.set("1, 2")
        p.subsets.set("1, 2")
        p.compare_cor.set(True)
        p._start()
        app._drain()
        assert not errors, errors
        assert p.history.get() == "Point"
        assert {row["case"] for row in p.results["Point"][0]} == {"Saved COR", "Estimated COR"}
        app.recon_panel.v["method"].set("mlem")
        app.recon_panel.v["iters"].set(1)
        app.recon_panel._recon_async()
        assert app.reconstructions[-1]["spacing"][2] == 1.68

        path = tmp_path / "coincidences.npz"
        phases = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        delta = np.column_stack((10 * np.cos(phases), 10 * np.sin(phases), 2 * np.sin(2 * phases)))
        np.savez(path, events=np.column_stack((-delta, np.full(32, 515), delta, np.full(32, 515))))
        panel = app.listmode_panel
        panel.path.set(str(path))
        panel.matrix.set("11 11 9")
        panel.voxel.set("1 1 1")
        panel.iterations.set(2)
        panel._start()
        app._drain()
        assert not errors, errors
        result = app.reconstructions[-1]
        assert result["source"] == "coincidences" and result["volume"].shape == (9, 11, 11)
        assert result["origin"] == (-5., -5., -4.)
        assert app.recon_workspace.select() == str(app.recon_panel)
    finally:
        app.destroy()
