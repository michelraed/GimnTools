"""Optional PySide6/VTK front end for sinogram, reconstruction and LUT workflows."""

from functools import partial

import numpy as np
from PySide6 import QtCore, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkInteractionImage import vtkImageViewer2
import vtkmodules.vtkInteractionStyle  # Registers rendering implementations.
import vtkmodules.vtkRenderingOpenGL2

from Gimntools.GimnRec.analytical_response import generate_analytical_lut, kernel_summary
from Gimntools.GimnRec.data_loaders import (
    get_default_parameters, load_detector_config, load_detectors_angles, load_rsectors,
)
from Gimntools.GimnRec.detector_response import DetectorResponse, ResponseSettings
from Gimntools.GimnRec.filters import make_psf_kernel
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.utils import _build_sitk_image, coincidence_to_lor


class _Signals(QtCore.QObject):
    ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class _Job(QtCore.QRunnable):
    def __init__(self, task, on_success):
        super().__init__()
        self.fn, self.on_success = task, on_success
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.ready.emit((self.on_success, self.fn()))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class SliceView(QtWidgets.QWidget):
    """VTK display for arrays in the package's (z, y, x) layout."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.canvas = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.canvas, 1)
        self.viewer = vtkImageViewer2()
        self.viewer.SetRenderWindow(self.canvas.GetRenderWindow())
        self.viewer.SetupInteractor(self.canvas.GetRenderWindow().GetInteractor())
        self.viewer.GetRenderer().SetBackground(0.06, 0.07, 0.09)
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.orientation = QtWidgets.QComboBox()
        self.orientation.addItems(["Axial", "Coronal", "Sagittal"])
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        controls.addWidget(self.orientation)
        controls.addWidget(self.slider, 1)
        self.position = QtWidgets.QLabel()
        controls.addWidget(self.position)
        self.orientation.currentIndexChanged.connect(self._orient)
        self.slider.valueChanged.connect(self._slice)
        self.image = None

    def show_array(self, volume, spacing=(1., 1., 1.)):
        image = vtkImageData()
        image.SetDimensions(*volume.shape[::-1])
        image.SetSpacing(*spacing)
        image.GetPointData().SetScalars(numpy_to_vtk(
            np.asarray(volume, dtype=np.float32, order="C").ravel(), deep=True))
        self.image = image
        self.viewer.SetInputData(image)
        low, high = float(volume.min()), float(volume.max())
        self.viewer.SetColorWindow(max(high - low, 1e-6))
        self.viewer.SetColorLevel((high + low) / 2)
        self.slider.setEnabled(True)
        self._orient(self.orientation.currentIndex())

    def _orient(self, selection):
        if self.image is None:
            return
        orient = (self.viewer.SetSliceOrientationToXY,
                  self.viewer.SetSliceOrientationToXZ,
                  self.viewer.SetSliceOrientationToYZ)[selection]
        orient()
        self.slider.setRange(self.viewer.GetSliceMin(), self.viewer.GetSliceMax())
        self.slider.setValue(self.viewer.GetSliceMax() // 2)
        self.viewer.GetRenderer().ResetCamera()
        self._slice(self.slider.value())

    def _slice(self, value):
        self.position.setText(str(value))
        self.viewer.SetSlice(value)
        self.viewer.Render()

    def closeEvent(self, event):
        self.canvas.Finalize()
        super().closeEvent(event)


def _field(form, label, default="", file=False):
    field = QtWidgets.QLineEdit(str(default))
    if not file:
        form.addRow(label, field)
        return field
    row = QtWidgets.QWidget()
    controls = QtWidgets.QHBoxLayout(row)
    controls.setContentsMargins(0, 0, 0, 0)
    controls.addWidget(field, 1)
    button = QtWidgets.QPushButton("Browse…")
    controls.addWidget(button)

    def choose():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(row, label, field.text(), file)
        if path:
            field.setText(path)

    button.clicked.connect(choose)
    form.addRow(label, row)
    return field


def _load_acquisition(paths):
    measurement, detector_config, orientations, rsectors, background, scale, threads = paths
    config = load_detector_config(detector_config)
    fmt = config["data_from"]
    angles = load_detectors_angles(orientations)
    mapping = load_rsectors(rsectors)
    params = get_default_parameters(fmt)
    metadata = {}
    sino, _ = coincidence_to_lor(measurement, config, angles, params, rsectors=mapping,
                                 geometry_out=metadata, workers=threads)
    estimate = None
    if background:
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Background scale must be nonnegative and finite")
        raw, _ = coincidence_to_lor(background, config, angles, params, rsectors=mapping, workers=threads)
        if raw.shape != sino.shape:
            raise ValueError("Measurement and background bins differ")
        estimate = scale * raw
        metadata.update(background=background, background_scale=scale)
    angular_grid = (np.arange(sino.shape[2]) + 0.5) * (180.0 / sino.shape[2])
    span = config.get("sino_range_mm", config["crystal_size"][1])
    return sino, config, angular_grid, span, estimate, metadata


def _reconstructor(acq, n, fov, dx=0, dy=0):
    sino, cfg, _angles, span, _bg, _metadata = acq
    return siddon_reconstructor(sino, fov_mm=fov or 2 * cfg["Half_FOV"],
                                sino_range_mm=span, n_img=n,
                                dx_cor_mm=dx, dy_cor_mm=dy, verbose=False)


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GimnTools · PET reconstruction")
        self.resize(1200, 760)
        self.acquisition = self.result = self.job = None
        self.pool = QtCore.QThreadPool(self)
        body = QtWidgets.QSplitter()
        self.setCentralWidget(body)
        self.tabs = QtWidgets.QTabWidget()
        body.addWidget(self.tabs)
        self.viewer = SliceView()
        body.addWidget(self.viewer)
        body.setStretchFactor(1, 1)
        body.setSizes([460, 740])
        self._acquisition_tab()
        self._reconstruction_tab()
        self._analytical_tab()

    def _tab(self, title):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, title)
        form = QtWidgets.QFormLayout(tab)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _acquisition_tab(self):
        form = self._tab("Acquisition")
        self.measurement = _field(form, "Measurement", file="ROOT (*.root)")
        self.detector_config = _field(form, "Detector setup", file="JSON (*.json)")
        self.detector_angles = _field(form, "Detector orientations", file="JSON (*.json)")
        self.chip_mapping = _field(form, "Chip/sector mapping", file="JSON (*.json)")
        self.background = _field(form, "Background (optional)", file="ROOT (*.root)")
        self.bg_scale = _field(form, "Background scale", 1)
        self.threads = _field(form, "PETSYS threads (0 auto)", 0)
        build = QtWidgets.QPushButton("Build sinogram")
        build.clicked.connect(self.build)
        form.addRow(build)
        show = QtWidgets.QPushButton("Show sinogram")
        show.clicked.connect(self._show_sino)
        form.addRow(show)

    def _reconstruction_tab(self):
        form = self._tab("Reconstruction")
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["OSEM", "MLEM", "FBP"])
        form.addRow("Method", self.method)
        self.response = QtWidgets.QComboBox()
        self.response.addItems(["Conventional", "Radial Gaussian", "DOI-aware",
                                "Spatially variant PSF", "DOI + residual SV response",
                                "Analytical detector response"])
        form.addRow("Detector response", self.response)
        self.n_img = _field(form, "Image matrix (pixels)", 64)
        self.fov = _field(form, "Image field (mm, blank uses detector setup)")
        self.iters = _field(form, "Iterations", 10)
        self.subsets = _field(form, "OSEM subsets", 4)
        self.fwhm = _field(form, "Radial Gaussian FWHM (mm)", 3)
        self.lut = _field(form, "SV-PSF LUT", file="NPZ (*.npz)")
        self.dx = _field(form, "COR X (mm)", 0)
        self.dy = _field(form, "COR Y (mm)", 0)
        self.encoder_offset = _field(form, "Encoder offset (°)", 0)
        run = QtWidgets.QPushButton("Reconstruct")
        run.clicked.connect(self.reconstruct)
        form.addRow(run)
        save = QtWidgets.QPushButton("Save volume…")
        save.clicked.connect(self.save)
        form.addRow(save)

    def _analytical_tab(self):
        form = self._tab("Analytical LUT")
        self.mu = _field(form, "511 keV μ (mm⁻¹)")
        self.face = _field(form, "Active segment width (mm)")
        self.face_order = _field(form, "Face quadrature", 3)
        self.depth_order = _field(form, "Depth quadrature", 3)
        self.mesh = _field(form, "Source samples per axis", 5)
        self.support = _field(form, "Kernel half width (samples)", 2)
        self.stride = _field(form, "Kernel step (voxels)", 1)
        self.regularization = _field(form, "Fit regularization", 0.0001)
        form.addRow(QtWidgets.QLabel("The generated full response includes DOI."))
        export = QtWidgets.QPushButton("Generate and save LUT…")
        export.clicked.connect(self.generate)
        form.addRow(export)

    def _run(self, label, task, done):
        if self.job is not None:
            return
        self.statusBar().showMessage(label)
        self.job = _Job(task, done)
        self.job.signals.ready.connect(self._ready)
        self.job.signals.failed.connect(self._failed)
        self.pool.start(self.job)

    @QtCore.Slot(object)
    def _ready(self, payload):
        self.job = None
        self.statusBar().clearMessage()
        callback, result = payload
        try:
            callback(result)
        except Exception as exc:
            self._failed(str(exc))

    @QtCore.Slot(str)
    def _failed(self, message):
        self.job = None
        self.statusBar().showMessage(message)
        QtWidgets.QMessageBox.warning(self, "GimnTools", message)

    def build(self):
        try:
            paths = tuple(widget.text().strip() for widget in (
                self.measurement, self.detector_config, self.detector_angles,
                self.chip_mapping, self.background,
            ))
            scale = float(self.bg_scale.text())
            threads = int(self.threads.text()) or None
            self._run("Building sinogram…",
                      partial(_load_acquisition, (*paths, scale, threads)), self._built)
        except ValueError as exc:
            self._failed(str(exc))

    def _built(self, result):
        self.acquisition, self.result = result, None
        self._show_sino()

    def _show_sino(self):
        if self.acquisition is not None:
            self.viewer.show_array(self.acquisition[0])

    def _response(self, *, doi=False, lut=False, analytical=False):
        path = self.lut.text().strip() if lut else None
        if lut and not path:
            raise ValueError("Select an SV-PSF LUT")
        return ResponseSettings(
            float(self.mu.text()) if doi else None, float(self.face.text()) if doi else None,
            int(self.face_order.text()) if doi else 3, int(self.depth_order.text()) if doi else 3,
            path, analytical=analytical,
        )

    def _grid(self):
        if self.acquisition is None:
            raise ValueError("Build a sinogram first")
        return (self.acquisition, int(self.n_img.text()), float(self.fov.text() or 0),
                float(self.dx.text()), float(self.dy.text()), float(self.encoder_offset.text()))

    def reconstruct(self):
        try:
            acq, n, fov, dx, dy, phi = self._grid()
            mode = self.response.currentIndex()
            method = self.method.currentText().lower()
            iters, subsets = int(self.iters.text()), int(self.subsets.text())
            fwhm = float(self.fwhm.text()) if mode == 1 else None
            response = self._response(doi=mode in (2, 4, 5), lut=mode in (3, 4),
                                      analytical=mode == 5) if method != "fbp" else None
        except ValueError as exc:
            self._failed(str(exc))
            return

        def run():
            sino, cfg, angles, span, bg, metadata = acq
            if method == "fbp" and bg is not None:
                sino = sino - bg
            angles = angles + phi
            recon = _reconstructor((sino, cfg, angles, span, bg, metadata), n, fov, dx, dy)
            model = DetectorResponse(recon, angles, response, metadata) if response and mode >= 2 else None
            if method == "fbp":
                volume = recon.fbp(angles)
            else:
                psf = make_psf_kernel(fwhm, recon.sino_pixel) if fwhm is not None else None
                options = dict(iterations=iters, psf_kernel=psf, background_sino=bg,
                               response=model, verbose=False)
                if method == "osem":
                    options["subsets_n"] = subsets
                volume = getattr(recon, method)(angles, **options)
            image, spacing = _build_sitk_image(volume, cfg, voxel_mm=recon.voxel_mm)
            if model is not None:
                import json
                image.SetMetaData("GimnToolsDetectorResponse", json.dumps(model.metadata))
            return volume, image, spacing

        self._run("Reconstructing…", run, self._reconstructed)

    def _reconstructed(self, result):
        volume, self.result, spacing = result
        self.viewer.show_array(volume, spacing)

    def save(self):
        if self.result is None:
            self._failed("Reconstruct a volume first")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save volume", "volume.nrrd",
                                                       "Image volumes (*.nrrd *.mhd)")
        if path:
            import SimpleITK as sitk
            try:
                sitk.WriteImage(self.result, path)
            except RuntimeError as exc:
                self._failed(str(exc))

    def generate(self):
        try:
            acq, n, fov, dx, dy, phi = self._grid()
            settings = self._response(doi=True)
            options = dict(sample_count=int(self.mesh.text()), half_support=int(self.support.text()),
                           kernel_stride=int(self.stride.text()), regularization=float(self.regularization.text()))
        except ValueError as exc:
            self._failed(str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save analytical LUT",
                                                       "analytical_lut.npz", "LUT (*.npz)")
        if path:
            self._run(
                "Generating analytical LUT…",
                lambda: generate_analytical_lut(
                    _reconstructor(acq, n, fov, dx, dy), acq[2] + phi, settings, acq[5], path, **options),
                partial(self._lut_ready, path),
            )

    def _lut_ready(self, path, result):
        self.lut.setText(path)
        self.response.setCurrentIndex(3)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Analytical LUT preview")
        dialog.resize(600, 590)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QtWidgets.QVBoxLayout(dialog)
        selector = QtWidgets.QComboBox()
        layout.addWidget(selector)
        view = SliceView()
        view.orientation.setEnabled(False)
        layout.addWidget(view, 1)
        metrics = QtWidgets.QLabel()
        metrics.setWordWrap(True)
        layout.addWidget(metrics)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        profiles = FigureCanvasQTAgg(Figure(figsize=(4, 2), layout="constrained"))
        axes = profiles.figure.subplots()
        layout.addWidget(profiles, 1)
        for y in result["y_mm"]:
            for x in result["x_mm"]:
                selector.addItem(f"x {x:.2f} mm, y {y:.2f} mm")

        def display(item):
            iy, ix = divmod(item, len(result["x_mm"]))
            summary = kernel_summary(result, iy, ix)
            view.show_array(result["kernels"][iy, ix][None])
            cx, cy = summary["centroid"]
            wx, wy = summary["fwhm"]
            error = result["relative_error"][iy, ix]
            extra = " · High residual, prefer projection-space response" if error > 0.30 else ""
            metrics.setText(
                f"Centroid ({cx:.2f}, {cy:.2f}) mm · FWHM ({wx:.2f}, {wy:.2f}) mm\n"
                f"Asymmetry {summary['asymmetry']:.3f} · Fit error {error:.1%} · "
                f"{result['elapsed_seconds']:.1f} s{extra}"
            )
            axes.clear()
            axes.plot(result["dx_mm"], summary["x_profile"], label="X")
            axes.plot(result["dy_mm"], summary["y_profile"], label="Y")
            axes.set_xlabel("Offset (mm)")
            axes.legend()
            profiles.draw_idle()

        selector.currentIndexChanged.connect(display)
        dialog.show()
        display(0)

    def closeEvent(self, event):
        if self.job is not None:
            self.statusBar().showMessage("Wait for the current operation to finish")
            event.ignore()
            return
        self.viewer.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Window()
    window.show()
    return app.exec()
