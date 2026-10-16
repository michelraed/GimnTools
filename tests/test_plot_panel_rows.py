"""Drive the multi-row Plot panel headlessly, with REAL reconstructions.

Requires a working Tk (the widgets are built for real, only the display is
virtual). Skipped when no display and no Xvfb is available.
"""
import os
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
tk = pytest.importorskip("tkinter")

try:
    _root = tk.Tk()
    _root.withdraw()
except Exception as exc:                                    # pragma: no cover
    pytest.skip(f"no usable Tk display: {exc}", allow_module_level=True)

from Gimntools.GimnRec.ui import app as uiapp                # noqa: E402
from Gimntools.GimnRec.reconstructor import siddon_reconstructor  # noqa: E402
from Gimntools.GimnRec.filters import make_fbp_filter        # noqa: E402

FOV, NR, NANG, NZ = 40.0, 64, 60, 3


@pytest.fixture(scope="module")
def data():
    """Two-blob phantom, exact projections, reconstructed by FBP and MLEM."""
    b = FOV / NR
    r = (np.arange(NR) + 0.5) * b - FOV / 2
    proj = np.zeros(NR)
    for c, s, a in ((-6.0, 2.0, 1.0), (5.0, 1.5, 0.6)):
        proj += a * np.sqrt(2 * np.pi) * s * np.exp(-(r - c) ** 2 / (2 * s ** 2))
    sino = np.repeat(np.repeat(proj[None, :, None], NANG, axis=2), NZ, axis=0)
    angles = np.linspace(0, 180, NANG, endpoint=False)
    rec = siddon_reconstructor(sino, fov_mm=FOV, sino_range_mm=FOV,
                               verbose=False)
    fbp = rec.fbp(angles, filter_type=make_fbp_filter("hann", cutoff=0.6))
    mlem = rec.mlem(angles, iterations=8, verbose=False)
    return sino, angles, fbp, mlem


class _FakeApp:
    """Minimal stand-in for ReconUI: only what PlotReconPanel touches."""

    def __init__(self, sino, angles, fbp, mlem):
        self.logs, self.errors = [], []
        self.sinograms = [{
            "name": "sino_A", "sino": sino, "angles": angles,
            "fov_mm": FOV, "sino_range_mm": FOV, "source": "GATE",
            "detector_config": None,
        }]
        sp = (FOV / fbp.shape[1],) * 3
        self.reconstructions = [
            {"name": "sino_A_fbp", "volume": fbp, "spacing": sp,
             "source": "sino_A", "method": "fbp", "info": "fbp"},
            {"name": "sino_A_mlem", "volume": mlem, "spacing": sp,
             "source": "sino_A", "method": "mlem", "info": "mlem"},
            {"name": "external", "volume": fbp, "spacing": sp,
             "source": "MHD", "method": "external", "info": "imported"},
        ]

    def get_sinogram(self, name):
        return next((s for s in self.sinograms if s["name"] == name), None)

    def log(self, m):
        self.logs.append(m)

    def error(self, m):
        self.errors.append(m)


@pytest.fixture
def panel(data):
    root = tk.Toplevel(_root)
    app = _FakeApp(*data)
    p = uiapp.PlotReconPanel(root, app)
    yield p
    root.destroy()


def _axes(panel):
    panel.update_preview()
    assert not panel.app.errors, panel.app.errors
    return panel.fig.axes


def test_empty_panel_renders_a_message_not_a_crash(panel):
    axs = _axes(panel)
    assert len(axs) == 1
    assert not axs[0].axison


def test_three_row_sinogram_fbp_mlem_layout(panel):
    """The headline case: sinogram / FBP / MLEM stacked in three rows."""
    panel.avail.selection_set(0)                       # the FBP entry
    panel._add_selected_as_columns()
    assert len(panel.columns) == 1
    assert len(panel._image_rows()) == 1

    panel.avail.selection_clear(0, tk.END)
    panel.avail.selection_set(1)                       # the MLEM entry
    panel._add_selected_as_row()
    assert len(panel._image_rows()) == 2

    srow = panel._add_row("sinogram")
    panel._move_row(srow, -1)                          # sinogram to the top
    panel._move_row(srow, -1)
    assert panel.rows[0] is srow

    kinds = [r["kind"].get() for r in panel.rows]
    assert kinds == ["sinogram", "image", "image"]

    axs = _axes(panel)
    # 3 rows x 1 column = 3 image axes (+ 3 colorbar axes by default)
    assert len(panel.fig.axes) == 6
    assert all(ax.axison for ax in axs[:3])


def test_sinogram_row_follows_the_reconstruction_source(panel):
    """The sinogram is derived from the recon's 'source', never chosen."""
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    srow = panel._add_row("sinogram")
    got = panel._sino_for(srow, panel.columns[0])
    assert got is not None
    assert got[3] == "sino_A"                          # resolved source name


def test_imported_volume_has_no_sinogram_and_says_so(panel):
    """An MHD import has source='MHD', which is not a sinogram: no panel."""
    panel.avail.selection_set(2)                       # the external entry
    panel._add_selected_as_columns()
    srow = panel._add_row("sinogram")
    assert panel._sino_for(srow, panel.columns[0]) is None
    axs = _axes(panel)
    texts = [t.get_text() for ax in axs for t in ax.texts]
    assert any("no sinogram" in t for t in texts)


def test_row_supertitles_are_rendered(panel):
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    panel.rows[0]["title"].set("FBP (Hann 0.6)")
    panel.avail.selection_clear(0, tk.END)
    panel.avail.selection_set(1)
    panel._add_selected_as_row()
    panel.rows[1]["title"].set("MLEM (8 it)")
    axs = _axes(panel)
    texts = [t.get_text() for ax in axs for t in ax.texts]
    assert "FBP (Hann 0.6)" in texts
    assert "MLEM (8 it)" in texts


def test_row_supertitle_placement_and_off(panel):
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    panel.rows[0]["title"].set("ROWLABEL")
    for pos in ("left", "above"):
        panel.v_row_title_pos.set(pos)
        texts = [t.get_text() for ax in _axes(panel) for t in ax.texts]
        assert "ROWLABEL" in texts, pos
    panel.v_row_title_pos.set("off")
    texts = [t.get_text() for ax in _axes(panel) for t in ax.texts]
    assert "ROWLABEL" not in texts


def test_per_column_sinogram_vmin_vmax(panel):
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    srow = panel._add_row("sinogram")
    panel.columns[0]["sino_vmin"].set("5")
    panel.columns[0]["sino_vmax"].set("40")
    _axes(panel)
    sino_ax = panel.fig.axes[0] if panel.rows[0] is srow else panel.fig.axes[2]
    im = sino_ax.images[0]
    assert im.get_clim() == (5.0, 40.0)


def test_cell_vmin_overrides_the_row_default(panel):
    panel.avail.selection_set(0)
    panel.avail.selection_set(1)
    panel._add_selected_as_columns()
    row = panel._image_rows()[0]
    row["vmin"].set("0"); row["vmax"].set("2")
    row["cells"][1]["vmax"].set("9")                   # override column 2 only
    _axes(panel)
    clims = [ax.images[0].get_clim() for ax in panel.fig.axes if ax.images]
    assert (0.0, 2.0) in clims
    assert (0.0, 9.0) in clims


def test_shared_in_row_uses_the_displayed_slices_only(panel):
    panel.avail.selection_set(0)
    panel.avail.selection_set(1)
    panel._add_selected_as_columns()
    row = panel._image_rows()[0]
    row["scale"].set("shared in row")
    _axes(panel)
    clims = {ax.images[0].get_clim() for ax in panel.fig.axes if ax.images}
    assert len(clims) == 1, f"shared scale not applied: {clims}"


def test_columns_share_one_slice_across_rows(panel):
    """A column fixes plane and slice once, for every row beneath it."""
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    panel.avail.selection_clear(0, tk.END)
    panel.avail.selection_set(1)
    panel._add_selected_as_row()
    col = panel.columns[0]
    assert panel._n_slices(col) == NZ
    col["slice"].set(2)
    _axes(panel)
    imgs = [ax.images[0].get_array() for ax in panel.fig.axes if ax.images]
    assert len(imgs) == 2
    # Both rows must show slice 2 of their own volume, same geometry.
    assert imgs[0].shape == imgs[1].shape


def test_moving_a_column_moves_its_cells(panel):
    panel.avail.selection_set(0)
    panel.avail.selection_set(1)
    panel._add_selected_as_columns()
    row = panel._image_rows()[0]
    before = [c["rec"].get() for c in row["cells"]]
    panel._move_column(panel.columns[0], +1)
    after = [c["rec"].get() for c in row["cells"]]
    assert after == before[::-1]


def test_removing_a_column_removes_its_cells(panel):
    panel.avail.selection_set(0)
    panel.avail.selection_set(1)
    panel._add_selected_as_columns()
    row = panel._image_rows()[0]
    assert len(row["cells"]) == 2
    panel._remove_column(panel.columns[0])
    assert len(panel.columns) == 1
    assert len(row["cells"]) == 1
    _axes(panel)


def test_coronal_falls_back_to_a_summed_sinogram(panel):
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    srow = panel._add_row("sinogram")
    panel.columns[0]["plane"].set("Coronal")
    note = panel._sino_for(srow, panel.columns[0])[2]
    assert "summed" in note


def test_save_figure_writes_a_real_file(panel, tmp_path):
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    panel._add_row("sinogram")
    out = tmp_path / "fig.png"
    panel.v_fmt.set("png")
    uiapp.filedialog.asksaveasfilename = lambda **k: str(out)
    uiapp.messagebox.showinfo = lambda *a, **k: None
    panel.save_figure()
    assert out.exists() and out.stat().st_size > 5000


def test_slice_slider_range_follows_the_assigned_volume(panel):
    """Adding a column must bound its slider by the volume it will show.

    Regression: the slider was configured at column creation, before the
    cell was filled, so it stayed clamped at 0 and the sinogram row silently
    kept showing slice 0 whatever the user selected.
    """
    panel.avail.selection_set(0)
    panel._add_selected_as_columns()
    col = panel.columns[0]
    assert float(col["scale_w"].cget("to")) == NZ - 1
    col["slice"].set(NZ - 1)
    assert col["slice"].get() == NZ - 1
    srow = panel._add_row("sinogram")
    assert panel._sino_for(srow, col)[2] == f"slice {NZ - 1}"
