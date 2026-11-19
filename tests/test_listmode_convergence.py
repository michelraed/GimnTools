"""Coincidence geometry, robust convergence and axial units regressions."""

import numpy as np
import pytest

from Gimntools.GimnRec.convergence import SourceRegion, convergence_series, detect_source_regions
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


def test_background_cv_classical_and_winsorized_use_the_same_roi():
    background = np.r_[np.linspace(1., 3., 19), 40.]
    image = np.r_[50., background][None, :]
    region = SourceRegion((0, 0, 0), (0,), (np.array([0]),), (np.arange(1, image.size),))

    def run(iterations, subsets, checkpoint):
        checkpoint(0, iterations, image)

    row, = convergence_series(run, [region], [1], [1], cor_mm=None, label="Point")
    winsor = np.clip(background, *np.quantile(background, [.05, .95]))
    assert row["background_voxels"] == background.size
    assert row["background_raw_mean"] == pytest.approx(background.mean())
    assert row["background_raw_sd"] == pytest.approx(background.std(ddof=1))
    assert row["background_cv_pct"] == pytest.approx(100 * background.std(ddof=1) / background.mean())
    assert row["noise_pct"] == pytest.approx(100 * winsor.std(ddof=1) / winsor.mean())
    assert row["noise_pct"] < row["background_cv_pct"]
    assert row["cnr"] == pytest.approx((50. - winsor.mean()) / winsor.std(ddof=1))


def test_background_roi_stays_away_from_all_source_rois_in_mm():
    spacing = (0.8, 1.1, 2.0)
    z, y, x = np.ogrid[:15, :31, :41]
    sources = ((7, 16, 16), (7, 16, 24))
    volume = sum(np.exp(-((z - zi)**2 + (y - yi)**2 + (x - xi)**2) / 2)
                 for zi, yi, xi in sources)
    regions = detect_source_regions(volume, spacing, radius_mm=2., background_gap_mm=4., source_count=2)
    assert {region.center_zyx for region in regions} == set(sources)
    for region in regions:
        for k, indices in zip(region.axial, region.background):
            coords = np.column_stack((np.full(indices.size, k), *np.unravel_index(indices, volume.shape[1:])))
            for center in sources:
                distance = np.linalg.norm((coords[:, 1:] - center[1:]) * np.array(spacing[1::-1]), axis=1)
                assert np.all(distance >= 6. - 1e-12)
        assert region.background_gap_mm == 4.
    with pytest.raises(ValueError, match="Fewer than 8 background voxels"):
        detect_source_regions(volume, spacing, radius_mm=2., background_gap_mm=60.)


@pytest.mark.parametrize("background", [[], [2.], [0., 0.]])
def test_background_cv_is_undefined_without_two_voxels_and_a_positive_mean(background):
    image = np.array([[5., *background]])
    region = SourceRegion((0, 0, 0), (0,), (np.array([0]),), (np.arange(1, image.size),))

    def run(iterations, subsets, checkpoint):
        checkpoint(0, iterations, image)

    row, = convergence_series(run, [region], [1], [1], cor_mm=None, label="Point")
    assert np.isnan(row["noise_pct"]) and np.isnan(row["background_cv_pct"])


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


def test_history_preserves_subset_volumes_written_by_slice():
    from Gimntools.GimnRec.iteration_history import IterationHistory

    history = IterationHistory()
    try:
        shape = (3, 4, 5)
        with history.record("Saved COR", 2, 3, shape=shape, spacing=(1., 1., 2.)) as append:
            for z in range(shape[0]):
                for iteration in range(1, 4):
                    for subset in range(1, 3):
                        append(z, iteration, subset, np.full(shape[1:], 10 * iteration + 2 * subset + z))
        series = history.runs["Saved COR", 2]
        np.testing.assert_array_equal(series.read(2, 1)[:, 0, 0], [22, 23, 24])
        np.testing.assert_array_equal(series.read(3, 2)[:, 0, 0], [34, 35, 36])
        with pytest.raises(ValueError, match="outside"):
            series.read(2, 3)
    finally:
        history.close()
    assert not history.directory.exists()


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
        p.matrix.set("23")
        p.voxel.set("0.9")
        p.compare_cor.set(True)
        p._start()
        app._drain()
        assert not errors, errors
        assert p.history.get() == "Point"
        assert {row["case"] for row in p.results["Point"][0]} == {"Saved COR", "Estimated COR"}
        history = app.iteration_panel.histories["Point"]
        steps = history.runs["Saved COR", 2]
        assert steps.shape == (5, 23, 23) and steps.spacing == (0.9, 0.9, 1.68)
        assert not np.array_equal(steps.read(1, 1), steps.read(1, 2))
        viewer = app.iteration_panel
        viewer.mode.set("Source ROIs")
        p._open_history()
        assert app.recon_workspace.select() == str(viewer)
        assert len(viewer.figure.axes[0].collections) == 2
        roi_image = tmp_path / "roi.pdf"
        monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **_: str(roi_image))
        viewer._save()
        assert roi_image.read_bytes().startswith(b"%PDF")
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
        panel.psf_mode.set("3D Gaussian field")
        panel._start()
        app._drain()
        assert not errors, errors
        result = app.reconstructions[-1]
        assert result["source"] == "coincidences" and result["volume"].shape == (9, 11, 11)
        assert result["origin"] == (-5., -5., -4.)
        assert result["geometry"]["sv_psf"] == "3D Gaussian SV-PSF"
        assert app.recon_workspace.select() == str(app.recon_panel)

        p.source.set(p.LISTMODE)
        p.compare_cor.set(False)
        p.matrix.set("13 11 9")
        p.voxel.set("0.9 1.1 1.2")
        p._start()
        app._drain()
        assert not errors, errors
        steps = viewer.histories["coincidences"].runs["List mode", 2]
        assert steps.read(2, 2).shape == (9, 11, 13)
        assert steps.spacing == (0.9, 1.1, 1.2)
        from csv import DictReader
        from types import SimpleNamespace
        from Gimntools.GimnRec.ui import em_panels

        p.noise_metric.set("Both")
        monkeypatch.setattr(em_panels, "BatchSelection",
                            lambda _parent, entries, **_kwargs: SimpleNamespace(result=entries))
        p._compare_results()
        assert p.comparison == ("Point", "coincidences")
        axes = np.asarray(p.figure.axes).reshape(4, 2)
        assert [axis.get_title() for axis in axes[0]] == ["Point", "coincidences"]
        assert [axis.get_ylabel() for axis in axes[:, 0]] == [
            "Contrast (%)", "CNR", "Robust background CV (%)", "Classical background CV (%)"
        ]
        assert axes[2, 0].get_shared_y_axes().joined(*axes[2, :])
        data = p.results["Point"][0]
        signal = [row for row in data if row["case"] == "Saved COR" and row["source"] == 1 and row["subsets"] == 1]
        np.testing.assert_allclose(axes[3, 0].lines[0].get_ydata(), [row["background_cv_pct"] for row in signal])
        csv_path = tmp_path / "comparison.csv"
        monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **_: str(csv_path))
        p._save_csv()
        with csv_path.open() as stream:
            rows = list(DictReader(stream))
        assert {row["acquisition"] for row in rows} == {"Point", "coincidences"}
        assert {"noise_pct", "background_cv_pct"}.issubset(rows[0])
        figure_path = tmp_path / "comparison.pdf"
        monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **_: str(figure_path))
        p._save_figure()
        assert figure_path.read_bytes().startswith(b"%PDF")
        p.history.set("Point")
        p._select_single_result()
        assert not p.comparison and len(p.figure.axes) == 3

        p.matrix.set("21")
        p.voxel.set("1")
        p.subsets.set("1")
        p.iterations.set("1")
        p.record_steps.set(False)
        p._start_batch()
        app._drain()
        assert not errors, errors
        assert len(p.comparison) == 2
        assert p.comparison == ("Point_2", "coincidences_2")
        assert len(p.figure.axes) == 8
    finally:
        app.destroy()
