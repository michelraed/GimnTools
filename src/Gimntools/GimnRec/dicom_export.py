"""Export an uncalibrated research reconstruction as a DICOM PET series."""

from datetime import datetime
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


DEFAULT_EQUIPMENT = "GIMN_PET"


def _description(value, name):
    value = str(value).strip()
    if not value or len(value) > 64 or any(c == "\\" or ord(c) < 32 for c in value):
        raise ValueError(f"{name} must contain 1–64 characters without backslashes or control characters")
    return value


def export_dicom_series(volume, spacing, destination, *, study_name, series_name,
                        equipment_name=DEFAULT_EQUIPMENT, origin=None, direction=None):
    """Write single-frame PET instances into a new directory within destination.

    The volume uses (z, y, x) array order and spacing (x, y, z) in mm.
    Output uses one common unsigned 16-bit linear rescale for the series.
    PET intensities are unitless and the export supplies no patient identity.
    """
    try:
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, PositronEmissionTomographyImageStorage, generate_uid
        from pydicom.valuerep import format_number_as_ds
    except ModuleNotFoundError as exc:
        raise ValueError("DICOM export requires pydicom. Install the updated GimnTools dependencies") from exc

    study = _description(study_name, "Study name")
    equipment = _description(equipment_name, "Equipment name")
    series = _description(str(series_name)[:64], "Reconstruction name")
    image = np.asarray(volume, dtype=float)
    if (image.ndim != 3 or not image.size or not np.isfinite(image).all()
            or max(image.shape) > 65535):
        raise ValueError("DICOM requires a nonempty, finite 3D reconstruction")
    nz, ny, nx = image.shape
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("DICOM requires three positive voxel spacings in mm")
    direction = np.eye(3) if direction is None else np.asarray(direction, dtype=float).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
        raise ValueError("DICOM orientation must be orthonormal")
    origin = (direction @ (-(np.array([nx, ny, nz]) - 1) * spacing / 2)
              if origin is None else np.asarray(origin, dtype=float))
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("DICOM requires a finite 3D image origin")
    parent = Path(destination)
    if not parent.is_dir():
        raise ValueError("Select an existing destination folder")

    low, high = float(image.min()), float(image.max())
    width = high - low
    if not np.isfinite(width):
        raise ValueError("DICOM intensity range is too wide to rescale")
    slope = width / 65535 if width else 1.0
    if slope == 0:
        raise ValueError("DICOM intensity range is too small to rescale")
    slope_ds, intercept_ds = map(format_number_as_ds, (slope, low))
    orientation = [format_number_as_ds(float(value)) for value in direction[:, :2].T.ravel()]
    study_uid, series_uid, frame_uid = (generate_uid() for _ in range(3))
    now = datetime.now()
    prefix = re.sub(r"[^\w-]+", "_", series).strip("_")[:40] or "PET"
    folder = Path(tempfile.mkdtemp(dir=parent, prefix=f"{prefix}-"))
    try:
        for z, plane in enumerate(image):
            uid = generate_uid()
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = PositronEmissionTomographyImageStorage
            meta.MediaStorageSOPInstanceUID = uid
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            path = folder / f"slice-{z + 1:05d}.dcm"
            ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SpecificCharacterSet = "ISO_IR 192"
            ds.SOPClassUID = PositronEmissionTomographyImageStorage
            ds.SOPInstanceUID = uid
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.Modality = "PT"
            ds.ImageType = ["DERIVED", "PRIMARY"]
            ds.StudyDescription = study
            ds.SeriesDescription = series
            ds.Manufacturer = ""
            ds.ManufacturerModelName = equipment
            if len(equipment) <= 16:
                ds.StationName = equipment
            ds.PatientName = ds.PatientID = ds.PatientBirthDate = ds.PatientSex = ""
            ds.StudyDate = ds.StudyTime = ds.AccessionNumber = ds.ReferringPhysicianName = ""
            ds.PatientOrientationCodeSequence = []
            ds.PatientGantryRelationshipCodeSequence = []
            ds.PositionReferenceIndicator = ""
            ds.StudyID = study_uid.rsplit(".", 1)[-1][-12:]
            ds.SeriesNumber = 1
            ds.Laterality = ""
            ds.InstanceNumber = ds.ImageIndex = z + 1
            ds.ContentDate, ds.ContentTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
            ds.AcquisitionDate = ds.AcquisitionTime = ""
            ds.SeriesType = ["STATIC", "IMAGE"]
            ds.Units = "NONE"
            ds.CountsSource = "EMISSION"
            ds.DecayCorrection = "NONE"
            ds.CollimatorType = ds.CorrectedImage = ""
            ds.FrameReferenceTime = 0
            ds.ActualFrameDuration = None
            ds.RadiopharmaceuticalInformationSequence = []
            ds.NumberOfSlices = nz
            ds.SeriesDate, ds.SeriesTime = ds.ContentDate, ds.ContentTime
            ds.ImageOrientationPatient = orientation
            position = origin + z * spacing[2] * direction[:, 2]
            ds.ImagePositionPatient = [format_number_as_ds(float(value)) for value in position]
            ds.PixelSpacing = [format_number_as_ds(float(value)) for value in spacing[[1, 0]]]
            ds.SliceThickness = ds.SpacingBetweenSlices = format_number_as_ds(float(spacing[2]))
            ds.Rows, ds.Columns = ny, nx
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.RescaleSlope, ds.RescaleIntercept = slope_ds, intercept_ds
            ds.PixelData = np.rint((plane - low) / slope).clip(0, 65535).astype("<u2").tobytes()
            ds.save_as(path, enforce_file_format=True)
    except Exception:
        shutil.rmtree(folder)
        raise
    return folder
