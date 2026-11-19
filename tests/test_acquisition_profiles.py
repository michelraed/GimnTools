"""SQLite detector format and material defaults in the shared GUI."""

import json
from pathlib import Path

import numpy as np
import pytest


def test_load_saved_profiles_only_into_the_matching_format(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from Gimntools.GimnRec.storage import Store
    from Gimntools.GimnRec.ui.app import ReconUI

    path = tmp_path / "legacy.db"
    monkeypatch.setenv("GIMNTOOLS_DB", str(path))
    examples = Path(__file__).resolve().parents[1] / "examples"
    pet = json.loads((examples / "CONFIG_PETSYS/detector_config_petsys.json").read_text())
    gate = json.loads((examples / "CONFIG_GATE/detector_config_gate.json").read_text())
    gate.update(crystal_material="BGO", slice_thickness_mm=1.5)
    db = Store()
    angles = {"left": 0, "right": -180}
    db.save_profile("PETSYS", fmt="PETSYS", detector_config=pet, angles=angles,
                    rsectors={0: "left", 15: "right"})
    db.save_profile("GATE_Simu_DOC", fmt="GATE", detector_config=gate, angles=angles,
                    rsectors={0: "left", 1: "right"})
    db.save_profile("PETSYS_2", fmt="petsys",
                    detector_config=dict(pet, radius=37, crystal_material="BGO"), angles=angles,
                    rsectors={0: "left", 15: "right"}, calibration={"dx_cor_mm": 0.4})

    try:
        app = ReconUI()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        pet_panel, gate_panel = app.petsys, app.gate
        assert pet_panel.profile.selected.get() == "PETSYS"
        assert gate_panel.profile.selected.get() == "GATE_Simu_DOC"
        assert set(pet_panel.profile.combo["values"]) == {"PETSYS", "PETSYS_2"}
        assert tuple(gate_panel.profile.combo["values"]) == ("GATE_Simu_DOC",)
        with pytest.raises(ValueError, match="matching tab"):
            app.load_acquisition_profile("PETSYS", expected="gate")
        assert gate_panel.cfg_form.get_config()["slice_thickness_mm"] == 1.5
        assert gate_panel.angles.get_rsectors() == {0: "left", 1: "right"}
        assert pet_panel.angles.get_rsectors() == {0: "left", 15: "right"}

        app.load_acquisition_profile("PETSYS_2", expected="petsys")
        assert pet_panel.calibration == {"dx_cor_mm": 0.4}
        assert pet_panel.var_apply_corr.get()
        assert db.get_setting("last_profile_petsys") == "PETSYS_2"
        app.load_acquisition_profile("PETSYS", expected="petsys")
        assert pet_panel.calibration is None
        assert not pet_panel.var_apply_corr.get()

        panel = app.recon_panel
        for name, tab, material, mu in (("gate", gate_panel, "BGO", 0.09629),
                                         ("petsys", pet_panel, "LYSO", 0.08184)):
            cfg = tab.cfg_form.get_config()
            app.set_sinogram(np.ones((1, 3, 6)), fov_mm=70, sino_range_mm=26.88,
                             source=name.upper(), name=name, detector_config=cfg)
            panel.refresh_sino_list(select=name)
            assert panel.v["doi_material"].get() == material
            assert float(panel.v["doi_mu"].get()) == pytest.approx(mu)
    finally:
        app.destroy()
