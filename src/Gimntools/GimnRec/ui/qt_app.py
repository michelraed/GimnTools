"""VTK slice rendering for the shared desktop workspace."""

from functools import partial
from weakref import ref

import numpy as np
from matplotlib.image import AxesImage
from PySide6.QtGui import QImage
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkImagingCore import vtkImageMapToColors


class VTKImageRenderer:
    """Use VTK to colorize scalar slices and Qt to carry the RGBA frame.

    The shared Matplotlib artist keeps the scalar image, normalization,
    coordinate transforms, overlays and colorbar.
    """

    def attach(self, artist):
        artist.to_rgba = partial(self._rgba, ref(artist))

    def _rgba(self, artist_ref, values, alpha=None, bytes=False, norm=True):
        artist = artist_ref()
        if np.ndim(values) != 2 or not np.size(values) or alpha is not None:
            return AxesImage.to_rgba(artist, values, alpha=alpha, bytes=bytes, norm=norm)
        values = np.ma.asarray(artist.norm(values) if norm else values)
        cmap = artist.get_cmap()
        size = cmap.N
        samples = values.data
        bad = values.mask if np.ma.is_masked(values) else np.isnan(samples)
        if np.issubdtype(samples.dtype, np.floating):
            samples = samples * size
            samples[samples == size] = size - 1
        under = samples < 0
        over = samples >= size
        index = np.where(np.isnan(samples), 0, samples).clip(0, size - 1).astype(np.int32)
        index[under] = size
        index[over] = size + 1
        index[bad] = size + 2

        precise = np.vstack((cmap(np.arange(size)), cmap(np.array([-1, size])),
                             cmap(np.ma.masked_all(1))))
        palette = (precise * 255).astype(np.uint8)
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(len(palette))
        lut.SetTableRange(0, len(palette) - 1)
        lut.SetTable(numpy_to_vtk(np.ascontiguousarray(palette), deep=True))

        rows, cols = index.shape
        image = vtkImageData()
        image.SetDimensions(cols, rows, 1)
        image.GetPointData().SetScalars(numpy_to_vtk(index.ravel(), deep=False))
        colors = vtkImageMapToColors()
        colors.SetInputData(image)
        colors.SetLookupTable(lut)
        colors.SetOutputFormatToRGBA()
        colors.Update()

        buffer = vtk_to_numpy(colors.GetOutput().GetPointData().GetScalars())
        frame = QImage(buffer.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888)
        # Keep the frame independent of the VTK buffer after returning.
        rgba = np.frombuffer(frame.constBits(), dtype=np.uint8).reshape(rows, cols, 4).copy()
        if bytes:
            return rgba
        # Restore the fractional palette values for Matplotlib's RGBA interpolation.
        return rgba / 255 + (precise - palette / 255)[index]


def main():
    from Gimntools.GimnRec.ui.app import main as run_desktop

    return run_desktop(image_renderer=VTKImageRenderer())
