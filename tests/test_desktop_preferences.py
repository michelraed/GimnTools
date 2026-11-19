"""Session settings carry across launchers and keep folders per workflow."""

import pytest

from Gimntools.GimnRec.storage import Store


def _open_app(*, themed):
    tk = pytest.importorskip("tkinter")
    from Gimntools.GimnRec.ui.app import ReconUI

    try:
        return ReconUI(themed=themed)
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")


def _font_picker(app):
    from tkinter import ttk

    widgets = app.winfo_children()
    while widgets:
        widget = widgets.pop()
        if (isinstance(widget, ttk.Combobox)
                and str(widget.cget("textvariable")) == str(app.typography.selection)):
            return widget
        widgets.extend(widget.winfo_children())
    pytest.fail("UI font selector is missing")


def test_switch_font_restore_controls_and_remember_dialog_folders(tmp_path, monkeypatch):
    from tkinter import filedialog, font

    db_path = tmp_path / "prefs.db"
    monkeypatch.setenv("GIMNTOOLS_DB", str(db_path))
    source_dir = tmp_path / "acquisitions"
    export_dir = tmp_path / "plots"
    source_dir.mkdir()
    export_dir.mkdir()
    root_path = source_dir / "legacy.root"
    events = source_dir / "events.npz"
    store = Store(db_path)
    store.set_setting("last_files", {"gate_root": str(root_path)})

    classic = _open_app(themed=False)
    try:
        assert classic.gate.var_root.get() == str(root_path)
        picker = _font_picker(classic)
        automatic = classic.typography.AUTO
        assert picker.get() == automatic
        old = font.nametofont("TkDefaultFont", root=classic).actual("family")
        alternatives = [name for name in picker.cget("values") if name != automatic and name.casefold() != old.casefold()]
        if not alternatives:
            pytest.skip("There is only one installed font")
        family = next((name for name in alternatives if name.casefold() == "courier"), alternatives[0])
        picker.set(family)
        picker.event_generate("<<ComboboxSelected>>")
        classic.update_idletasks()
        assert font.nametofont("TkDefaultFont", root=classic).cget("family").casefold() == family.casefold()

        monkeypatch.setattr(filedialog, "askopenfilename", lambda **_: str(events))
        classic.listmode_panel._browse()
        classic.session.choose("figure.save", lambda **_: str(export_dir / "figure.png"))
        classic.recon_panel.v["iters"].set(27)
        classic.recon_panel.v["iters"].set("")  # An interrupted edit must keep the last valid value.
        classic.listmode_panel.matrix.set("31 35 39")
        classic.listmode_panel.psf_mode.set("3D Gaussian field")
        classic.listmode_panel.psf_fwhm.set("1.3 1.8")
        classic.pseudo_panel.background.set(.75)
        classic.convergence_panel.subsets.set("1, 3, 6")
        classic.convergence_panel.matrix.set("63")
        classic.convergence_panel.noise_metric.set("Classical CV (%)")
        classic.sino_panel.ctrl.cmap.set("gray")
        classic.dicom_panel.equipment.set("GIMN_PET_2")
        classic.gate.cfg_form.vars["radius"].set("71.2")
        classic.gate.angles.set_data({"Left": 3., "Right": 183.}, {1: "Left", 8: "Right"})
        classic.notebook.select(classic.recon_workspace)
        classic.recon_workspace.select(classic.convergence_panel)
        classic.update()
    finally:
        classic.destroy()

    themed = _open_app(themed=True)
    try:
        picker = _font_picker(themed)
        assert picker.get() == family
        assert themed.typography.heading.cget("family").casefold() == family.casefold()
        assert themed.recon_panel.v["iters"].get() == 27
        assert themed.listmode_panel.path.get() == str(events)
        assert themed.listmode_panel.matrix.get() == "31 35 39"
        assert themed.listmode_panel.psf_mode.get() == "3D Gaussian field"
        assert themed.listmode_panel.psf_fwhm.get() == "1.3 1.8"
        assert themed.pseudo_panel.background.get() == .75
        assert themed.convergence_panel.subsets.get() == "1, 3, 6"
        assert themed.convergence_panel.matrix.get() == "63"
        assert themed.convergence_panel.noise_metric.get() == "Classical CV (%)"
        assert themed.sino_panel.ctrl.cmap.get() == "gray"
        assert themed.dicom_panel.equipment.get() == "GIMN_PET_2"
        assert themed.gate.cfg_form.get_config()["radius"] == 71.2
        assert themed.gate.angles.get_rsectors() == {1: "Left", 8: "Right"}
        assert themed.notebook.select() == str(themed.recon_workspace)
        assert themed.recon_workspace.select() == str(themed.convergence_panel)

        opened = []
        monkeypatch.setattr(filedialog, "askopenfilename",
                            lambda **opts: opened.append(opts["initialdir"]) or "")
        themed.listmode_panel._browse()
        themed.session.choose("figure.save", filedialog.askopenfilename)
        assert opened == list(map(str, (source_dir, export_dir)))
        assert themed.listmode_panel.path.get() == str(events)
        source_dir.rmdir()
        assert themed.session.directory("listmode.open") == str(export_dir)

        picker.set(automatic)
        picker.event_generate("<<ComboboxSelected>>")
        themed.update_idletasks()
        assert font.nametofont("TkDefaultFont", root=themed).actual("family") == themed.typography.default_family
    finally:
        themed.destroy()
    assert store.get_setting("desktop.ui.font") == automatic


def test_detector_drafts_follow_the_profile_revision(tmp_path, monkeypatch):
    db_path = tmp_path / "profiles.db"
    monkeypatch.setenv("GIMNTOOLS_DB", str(db_path))
    store = Store(db_path)
    store.save_profile("GATE", fmt="gate", detector_config={"data_from": "GATE", "radius": 39.})
    first = _open_app(themed=False)
    try:
        first.gate.cfg_form.vars["radius"].set("42")
    finally:
        first.destroy()
    assert store.get_profile("GATE")["detector_config"]["radius"] == 39.
    second = _open_app(themed=True)
    try:
        assert second.gate.cfg_form.get_config()["radius"] == 42.
    finally:
        second.destroy()

    store.save_profile("GATE", fmt="gate", detector_config={"data_from": "GATE", "radius": 46.})
    third = _open_app(themed=False)
    try:
        assert third.gate.cfg_form.get_config()["radius"] == 46.
    finally:
        third.destroy()
