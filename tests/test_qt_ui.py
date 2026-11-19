"""Rendering and SQLite parity between the GUI launchers."""

import os

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure

from test_detector_response import setup_geometry, write_root


def _vtk_renderer():
    pytest.importorskip("vtkmodules")
    pytest.importorskip("PySide6.QtGui")
    from Gimntools.GimnRec.ui.qt_app import VTKImageRenderer

    return VTKImageRenderer()


def test_vtk_preserves_scalar_colors_and_rendered_layout():
    from types import SimpleNamespace
    from Gimntools.GimnRec.ui.rendering import show_image

    renderer = _vtk_renderer()
    cmap = ListedColormap([(0.12345, 0.3629, 0.722, 1), (0.3, 0.8132, 0.624, 0.555)]).with_extremes(
        bad=to_rgba("#80ff40", 0.4), under="#bcd404", over="#9c028f")
    values = np.ma.array([[-1, 0, 0.499, 0.5], [1, 1.1, np.nan, 0.8]],
                         mask=[[False, False, True, False], [False, False, False, False]])
    images = []
    for backend in (None, renderer):
        figure = Figure(figsize=(3, 2))
        canvas = FigureCanvasAgg(figure)
        ax = figure.subplots()
        artist = show_image(SimpleNamespace(image_renderer=backend), ax, values, cmap=cmap,
                            origin="lower", vmin=0, vmax=1, interpolation="bilinear")
        assert artist.get_clim() == (0, 1)
        np.testing.assert_array_equal(artist.get_array(), np.ma.masked_invalid(values))
        assert tuple(artist.get_extent()) == (-0.5, 3.5, -0.5, 1.5)
        figure.colorbar(artist)
        canvas.draw()
        images.append(np.asarray(canvas.buffer_rgba()).copy())
        for as_bytes in (False, True):
            from matplotlib.image import AxesImage

            expected = AxesImage.to_rgba(artist, values, bytes=as_bytes)
            np.testing.assert_array_equal(artist.to_rgba(values, bytes=as_bytes), expected)
    np.testing.assert_array_equal(*images)


def test_tk_and_vtk_share_tabs_profiles_and_session(tmp_path, monkeypatch):
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("The desktop interface requires a display")
    renderer = _vtk_renderer()
    from Gimntools.GimnRec.ui.app import ReconUI

    db = tmp_path / "existing.db"
    monkeypatch.setenv("GIMNTOOLS_DB", str(db))
    root = tmp_path / "measurement.root"
    write_root(root, "PETSYS")
    config, angles, rsectors = setup_geometry("PETSYS")
    calibration = {"dx_cor_mm": 0.3, "dy_cor_mm": -0.2, "phi_offset_deg": 0.1}

    def tabs(app):
        return [app.notebook.tab(tab, "text") for tab in app.notebook.tabs()]

    tk = ReconUI()
    try:
        original_tabs = tabs(tk)
        tk.petsys.cfg_form._populate(config)
        tk.petsys.angles.set_data(angles, rsectors)
        tk.petsys.calibration = calibration
        tk.petsys.var_meas.set(str(root))
        tk.profiles_panel.var_name.set("saved-in-tk")
        tk.profiles_panel.var_pesq.set("researcher")
        tk.profiles_panel.var_exp.set("dual-head")
        tk.profiles_panel._save()
        tk._save_files()
    finally:
        tk.destroy()

    vtk = ReconUI(image_renderer=renderer)
    try:
        assert tabs(vtk) == original_tabs
        assert vtk.store.path == str(db)
        assert vtk.petsys.var_meas.get() == str(root)
        panel = vtk.profiles_panel
        panel.tree.selection_set(panel.tree.get_children()[0])
        panel._load()
        assert vtk.petsys.angles.get_angles() == angles
        assert vtk.petsys.angles.get_rsectors() == rsectors
        assert vtk.petsys.calibration == calibration
        assert vtk.petsys.cfg_form.get_config()["radius"] == config["radius"]
        assert panel.var_pesq.get() == "researcher"
        assert panel.var_exp.get() == "dual-head"

        errors = []
        vtk.error = errors.append
        vtk.start_worker = lambda _label, func, *args: func(*args)
        vtk.petsys._build_async()
        assert not errors, errors
        assert vtk.sinograms[-1]["sino"].sum() == 96
        vtk.sino_panel.draw()
        vtk.sino_panel.canvas.draw()
        image = vtk.sino_panel.ax.images[0]
        assert image.to_rgba.func.__self__ is renderer

        panel.var_name.set("saved-in-vtk")
        panel._save()
        vtk.petsys.var_meas.set("vtk-session.root")
        vtk._save_files()
    finally:
        vtk.destroy()

    tk = ReconUI()
    try:
        assert tk.store.path == str(db)
        assert tk.petsys.var_meas.get() == "vtk-session.root"
        panel = tk.profiles_panel
        profile = next(row for row in panel.tree.get_children()
                       if panel.tree.item(row, "values")[0] == "saved-in-vtk")
        panel.tree.selection_set(profile)
        panel._load()
        assert tk.petsys.calibration == calibration
        assert tk.petsys.angles.get_rsectors() == rsectors
    finally:
        tk.destroy()


def test_vtk_desktop_theme_applies_to_controls_and_plots_and_persists(tmp_path, monkeypatch):
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("The desktop interface requires a display")
    renderer = _vtk_renderer()
    from matplotlib.colors import to_rgba
    from tkinter import ttk
    from Gimntools.GimnRec.ui.app import ReconUI

    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "appearance.db"))
    app = ReconUI(image_renderer=renderer, themed=True)
    dark = app.appearance.palette
    tabs = app.notebook.tabs()
    try:
        assert app.appearance.mode.get() == "dark"
        assert app.log_text.cget("background") == dark["surface"]
        data = np.arange(42).reshape(1, 6, 7)
        app.set_sinogram(data, fov_mm=50, sino_range_mm=26.88, source="PETSYS")
        app.sino_panel.draw()
        app.sino_panel.canvas.draw()
        ax = app.sino_panel.ax
        assert ax.get_facecolor() == to_rgba(dark["surface"])

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        light_button = next(widget for widget in descendants(app)
                            if isinstance(widget, ttk.Radiobutton)
                            and widget.cget("style") == "Theme.TRadiobutton"
                            and widget.cget("value") == "light")
        light_button.invoke()
        app.sino_panel.canvas.draw()
        light = app.appearance.palette
        assert app.appearance.mode.get() == "light"
        assert light != dark
        assert app.log_text.cget("background") == light["surface"]
        assert ax.get_facecolor() == to_rgba(light["surface"])
        np.testing.assert_array_equal(ax.images[0].get_array(), data[0])
        assert app.notebook.tabs() == tabs
        assert app.store.get_setting("vtk_desktop_theme") == "light"
    finally:
        app.destroy()

    reopened = ReconUI(image_renderer=renderer, themed=True)
    try:
        assert reopened.appearance.mode.get() == "light"
    finally:
        reopened.destroy()
    classic = ReconUI()
    try:
        assert classic.appearance is None
        assert classic.store.get_setting("vtk_desktop_theme") == "light"
    finally:
        classic.destroy()
