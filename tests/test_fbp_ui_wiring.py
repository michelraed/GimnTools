"""Drive the real ReconUI.run_recon with the new FBP controls.

This is the part that unit tests on filters.py cannot cover: that the
widgets are actually read, passed down, and recorded on the reconstruction.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
tk = pytest.importorskip("tkinter")

try:
    _probe = tk.Tk(); _probe.destroy()
except Exception as exc:                                    # pragma: no cover
    pytest.skip(f"no usable Tk display: {exc}", allow_module_level=True)

from Gimntools.GimnRec.ui import app as uiapp                # noqa: E402

FOV, NR, NANG, NZ = 40.0, 64, 60, 2


@pytest.fixture(scope="module")
def ui():
    app = uiapp.ReconUI()
    app.withdraw()
    b = FOV / NR
    r = (np.arange(NR) + 0.5) * b - FOV / 2
    proj = np.sqrt(2 * np.pi) * 2.0 * np.exp(-r ** 2 / (2 * 2.0 ** 2))
    sino = np.repeat(np.repeat(proj[None, :, None], NANG, axis=2), NZ, axis=0)
    app.sinograms.append({
        "name": "S", "sino": sino, "fov_mm": FOV, "sino_range_mm": FOV,
        "angles": np.linspace(0, 180, NANG, endpoint=False),
        "source": "GATE", "detector_config": None, "norm": None,
    })
    yield app
    app.destroy()


def _run(ui, name, **overrides):
    v = ui.recon_panel.v
    v["method"].set("fbp")
    v["grid_mode"].set("sinogram")
    v["fbp_window"].set("ramp")
    v["fbp_cutoff"].set(1.0)
    v["fbp_order"].set(5)
    v["fbp_hard_cut"].set(True)
    v["fbp_pad_mode"].set("pow2")
    v["fbp_pad_factor"].set(2)
    v["fbp_interp"].set("none")
    v["fbp_interp_factor"].set(2)
    v["fbp_legacy_scale"].set(False)
    for k, val in overrides.items():
        v[k].set(val)
    before = len(ui.reconstructions)
    ui.run_recon(v, "S", name)
    assert len(ui.reconstructions) == before + 1, "reconstruction failed"
    return ui.reconstructions[-1]


def test_defaults_produce_a_quantitative_reconstruction(ui):
    rec = _run(ui, "r_default")
    assert rec["method"] == "fbp"
    assert rec["source"] == "S"           # what the sinogram row keys on
    assert rec["volume"].max() == pytest.approx(1.0, abs=0.05)


def test_fbp_settings_are_recorded_on_the_reconstruction(ui):
    rec = _run(ui, "r_rec", fbp_window="hann", fbp_cutoff=0.5,
               fbp_pad_mode="exact", fbp_pad_factor=3)
    f = rec["fbp"]
    assert f["window"] == "hann"
    assert f["cutoff_nyquist"] == pytest.approx(0.5)
    assert f["pad_mode"] == "exact" and f["pad_length"] == 3 * NR
    # cutoff in mm^-1 must match the radial bin actually used
    assert f["cutoff_mm_inv"] == pytest.approx(0.5 / (2 * FOV / NR))
    assert "hann" in rec["info"]


@pytest.mark.parametrize("window", list(uiapp.FBP_WINDOWS))
def test_every_window_runs(ui, window):
    rec = _run(ui, f"r_w_{window}", fbp_window=window, fbp_cutoff=0.6)
    assert np.isfinite(rec["volume"]).all()


@pytest.mark.parametrize("mode,factor", [("none", 1), ("none", 4),
                                         ("pow2", 2), ("exact", 3)])
def test_every_padding_procedure_runs(ui, mode, factor):
    rec = _run(ui, f"r_p_{mode}{factor}", fbp_pad_mode=mode,
               fbp_pad_factor=factor)
    assert rec["fbp"]["pad_length"] == uiapp.resolve_pad_length(NR, factor, mode)


def test_lower_cutoff_smooths_the_image(ui):
    """Empirical check that the control does what it claims."""
    sharp = _run(ui, "r_sharp", fbp_window="hann", fbp_cutoff=1.0)["volume"][0]
    smooth = _run(ui, "r_smooth", fbp_window="hann", fbp_cutoff=0.2)["volume"][0]
    grad = lambda a: np.abs(np.diff(a, axis=1)).mean()
    assert grad(smooth) < grad(sharp)


@pytest.mark.parametrize("kind", ["linear", "cubic", "conservative"])
def test_radial_interpolation_changes_the_matrix_in_sinogram_mode(ui, kind):
    """In 'sinogram' grid mode the matrix follows the bins — and we log it."""
    rec = _run(ui, f"r_i_{kind}", fbp_interp=kind, fbp_interp_factor=2)
    assert rec["volume"].shape[1] == 2 * NR
    assert rec["fbp"]["interp"] == kind and rec["fbp"]["interp_factor"] == 2


def test_interpolation_keeps_the_scale_on_a_fixed_grid(ui):
    """With the grid pinned, resampling the projections must NOT rescale the
    image. This is exactly what the corrected FBP constant buys."""
    base = _run(ui, "r_ni", grid_mode="matrix", n_img=64)["volume"].max()
    for k in (2, 4):
        got = _run(ui, f"r_i{k}", grid_mode="matrix", n_img=64,
                   fbp_interp="linear", fbp_interp_factor=k)["volume"].max()
        assert got == pytest.approx(base, rel=0.06), (
            f"interp x{k} rescaled the image: {got} vs {base}")


def test_legacy_scale_reproduces_the_old_ratio(ui):
    """On a volume-driven grid the legacy constant differs by voxel/bin."""
    new = _run(ui, "r_new", grid_mode="matrix", n_img=128)["volume"].max()
    old = _run(ui, "r_old", grid_mode="matrix", n_img=128,
               fbp_legacy_scale=True)["volume"].max()
    voxel_over_bin = (FOV / 128) / (FOV / NR)
    assert old / new == pytest.approx(voxel_over_bin, rel=1e-6)


def test_cutoff_hint_reports_physical_units(ui):
    ui.recon_panel.var_sino.set("S")
    ui.recon_panel.v["fbp_cutoff"].set(0.5)
    ui.recon_panel.v["fbp_interp"].set("none")
    ui.recon_panel._update_fbp_hint()
    txt = ui.recon_panel.lbl_cutoff.cget("text")
    nyq = 1.0 / (2.0 * FOV / NR)
    assert f"{0.5 * nyq:.3f} mm" in txt


def test_cutoff_hint_accounts_for_interpolation(ui):
    """Interpolating halves the bin, so Nyquist doubles — the hint must move."""
    p = ui.recon_panel
    p.var_sino.set("S")
    p.v["fbp_interp"].set("none")
    p._update_fbp_hint()
    a = p.lbl_cutoff.cget("text")
    p.v["fbp_interp"].set("linear")
    p.v["fbp_interp_factor"].set(2)
    p._update_fbp_hint()
    b = p.lbl_cutoff.cget("text")
    # FOV 40 mm / 64 bins = 0.6250 mm, halved to 0.3125 mm by the x2 resample
    assert a != b and "0.6250 mm" in a and "0.3125 mm" in b
