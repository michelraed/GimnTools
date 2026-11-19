"""Volumetric EM snapshots and the reference ROIs used for a sweep."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np


_DTYPE = np.dtype("<f4")


def grid_labels(shape, spacing):
    return {
        "matrix_xyz": " × ".join(map(str, shape[::-1])),
        "voxel_xyz_mm": " × ".join(f"{step:.4g}" for step in spacing),
    }


@dataclass(frozen=True)
class StepSeries:
    path: Path
    iterations: int
    subsets: int
    shape: tuple
    spacing: tuple

    def read(self, iteration, subset):
        if not 1 <= iteration <= self.iterations or not 1 <= subset <= self.subsets:
            raise ValueError("Iteration or subset is outside the recorded range")
        length = math.prod(self.shape)
        frame = (iteration - 1) * self.subsets + subset - 1
        volume = np.fromfile(self.path, dtype=_DTYPE, count=length,
                             offset=frame * length * _DTYPE.itemsize)
        return volume.reshape(self.shape)


class IterationHistory:
    def __init__(self):
        self._temp = TemporaryDirectory(prefix="gimntools-iterations-")
        self.directory = Path(self._temp.name)
        self.runs = {}
        self.reference = None
        self.regions = ()

    def set_reference(self, reference, spacing, regions):
        path = self.directory / "reference.raw"
        np.asarray(reference, dtype=_DTYPE).tofile(path)
        self.reference = StepSeries(path, 1, 1, reference.shape, tuple(spacing))
        self.regions = tuple(regions)

    @contextmanager
    def record(self, case, subsets, iterations, *, shape, spacing):
        """Write one slice at a time, or a complete 3D update for list mode."""
        path = self.directory / f"run-{len(self.runs)}.raw"
        slice_length = math.prod(shape[1:])
        required = math.prod(shape) * subsets * iterations * _DTYPE.itemsize
        if required > shutil.disk_usage(self.directory).free:
            raise ValueError(
                f"Iteration history needs {required / 1024**3:.2f} GiB for this setting. "
                "Free disk space or clear Record iteration images."
            )

        with path.open("wb", buffering=0) as stream:
            def write(z, iteration, subset, image):
                frame = (iteration - 1) * subsets + subset - 1
                start = (frame * shape[0] + (0 if z is None else z)) * slice_length
                stream.seek(start * _DTYPE.itemsize)
                np.asarray(image, dtype=_DTYPE).tofile(stream)

            yield write
        self.runs[case, subsets] = StepSeries(path, iterations, subsets, tuple(shape), tuple(spacing))

    def close(self):
        self._temp.cleanup()
