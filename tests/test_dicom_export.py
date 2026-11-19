"""DICOM series metadata, geometry and round-trip intensity checks."""

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.uid import PositronEmissionTomographyImageStorage

from Gimntools.GimnRec.dicom_export import export_dicom_series


def test_dicom_export_readable_geometry_and_value_rescale(tmp_path):
    values = np.linspace(-0.5, 5, 60).reshape(3, 4, 5)
    direction = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    origin = (-8.0, -10.0, -9.0)
    spacing = (0.84, 0.95, 3.36)
    folder = export_dicom_series(values, spacing, tmp_path,
                                 study_name="Estudo de duas fontes", series_name="OSEM 0.84 mm",
                                 origin=origin, direction=direction)
    files = sorted(folder.glob("*.dcm"))
    images = [pydicom.dcmread(path) for path in files]
    assert len(images) == 3
    assert {ds.Modality for ds in images} == {"PT"}
    assert {ds.SOPClassUID for ds in images} == {PositronEmissionTomographyImageStorage}
    assert {ds.StudyDescription for ds in images} == {"Estudo de duas fontes"}
    assert {ds.ManufacturerModelName for ds in images} == {"GIMN_PET"}
    assert {ds.StationName for ds in images} == {"GIMN_PET"}
    assert {ds.Units for ds in images} == {"NONE"}
    assert len({ds.StudyInstanceUID for ds in images}) == 1
    assert len({ds.SeriesInstanceUID for ds in images}) == 1
    assert len({ds.FrameOfReferenceUID for ds in images}) == 1
    assert len({ds.SOPInstanceUID for ds in images}) == 3
    for i, ds in enumerate(images):
        assert ds.ImageIndex == ds.InstanceNumber == i + 1
        np.testing.assert_allclose(ds.ImageOrientationPatient, [0, 1, 0, -1, 0, 0])
        np.testing.assert_allclose(ds.ImagePositionPatient, np.array(origin) + [0, 0, i*spacing[2]])
        np.testing.assert_allclose(ds.PixelSpacing, [spacing[1], spacing[0]])

    loaded = sitk.ReadImage([str(path) for path in files])
    np.testing.assert_allclose(loaded.GetDirection(), direction.ravel())
    np.testing.assert_allclose(loaded.GetOrigin(), origin)
    np.testing.assert_allclose(loaded.GetSpacing(), spacing)
    np.testing.assert_allclose(sitk.GetArrayFromImage(loaded), values,
                               atol=float(images[0].RescaleSlope)/2)


def test_dicom_export_preserves_constant_values_and_separates_runs(tmp_path):
    image = np.full((1, 2, 3), -4.0)
    args = (image, (1, 1, 3), tmp_path)
    opts = dict(study_name="Study test", series_name="MLEM", equipment_name="PET_BGO")
    folders = [export_dicom_series(*args, **opts) for _ in range(2)]
    assert folders[0] != folders[1]
    for folder in folders:
        ds = pydicom.dcmread(next(folder.glob("*.dcm")))
        assert ds.ManufacturerModelName == "PET_BGO"
        np.testing.assert_array_equal(ds.pixel_array * float(ds.RescaleSlope)
                                      + float(ds.RescaleIntercept), image[0])
    with pytest.raises(ValueError, match="Study name"):
        export_dicom_series(*args, **dict(opts, study_name=""))


def test_dicom_tab_exports_only_selected_volume(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from Gimntools.GimnRec.ui.app import ReconUI
    from Gimntools.GimnRec.ui import dicom_panel

    monkeypatch.setenv("GIMNTOOLS_DB", str(tmp_path / "ui.db"))
    try:
        app = ReconUI()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        panel = app.dicom_panel
        assert panel.equipment.get() == "GIMN_PET"
        assert "disabled" in panel.save.state()
        out = tmp_path / "exports"
        out.mkdir()
        app.add_reconstruction({"name": "coarse", "volume": np.ones((2, 4, 4)),
                                "spacing": (1.5, 1.5, 3.36)})
        app.add_reconstruction({"name": "fine", "volume": np.full((3, 6, 6), 2.0),
                                "spacing": (0.84, 0.84, 3.36)})
        panel.refresh()
        panel.reconstruction.set("fine")
        panel.study.set("Phantom axial")
        panel.equipment.set("GIMN_PET_BGO")
        monkeypatch.setattr(dicom_panel.filedialog, "askdirectory", lambda **_kw: str(out))

        def immediate(_label, fn):
            fn()
            return True

        monkeypatch.setattr(app, "start_worker", immediate)
        panel.save.invoke()
        app.after_cancel(app._drain_job)
        app._drain()
        files = sorted(out.glob("*/*.dcm"))
        assert len(files) == 3
        assert "Saved series" in panel.result.cget("text")
        first = pydicom.dcmread(files[0])
        assert first.SeriesDescription == "fine"
        assert first.StudyDescription == "Phantom axial"
        assert first.ManufacturerModelName == "GIMN_PET_BGO"
        assert first.Rows == first.Columns == 6
        np.testing.assert_allclose(first.PixelSpacing, [0.84, 0.84])
    finally:
        app.destroy()
