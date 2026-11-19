"""Smoke test for the optional Qt/VTK application on a graphical display."""

import json
import os
import time

import pytest

from test_detector_response import setup_geometry, write_root


def test_qt_vtk_builds_and_reconstructs(tmp_path):
    if not os.environ.get("DISPLAY"):
        pytest.skip("Qt/VTK requires a display")
    pytest.importorskip("vtkmodules")
    qt = pytest.importorskip("PySide6.QtWidgets")
    from Gimntools.GimnRec.ui.qt_app import Window

    app = qt.QApplication.instance() or qt.QApplication([])
    window = Window()
    window.show()
    errors = []
    window._failed = errors.append

    def wait():
        deadline = time.monotonic() + 30
        while window.job is not None and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not errors, errors
        assert window.job is None

    try:
        cfg, angles, rsectors = setup_geometry("PETSYS")
        for widget, name, data in ((window.detector_config, "config", cfg),
                                    (window.detector_angles, "angles", angles),
                                    (window.chip_mapping, "rsectors", rsectors)):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(data))
            widget.setText(str(path))
        root = tmp_path / "measurement.root"
        write_root(root, "PETSYS")
        window.measurement.setText(str(root))
        window.build()
        wait()
        assert window.acquisition[0].sum() == 96
        assert window.viewer.image.GetDimensions() == (42, 7, 7)
        window.method.setCurrentText("MLEM")
        window.iters.setText("1")
        window.n_img.setText("12")
        window.reconstruct()
        wait()
        assert window.result.GetSize() == (12, 12, 7)
        window.viewer.orientation.setCurrentText("Sagittal")
        assert window.viewer.slider.maximum() == 11
    finally:
        window.pool.waitForDone(30000)
        window.close()
