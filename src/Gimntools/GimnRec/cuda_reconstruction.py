"""CUDA kernels for the shared Siddon and sparse reconstruction operators."""

from collections import OrderedDict
from math import isfinite
from weakref import ref

import numpy as np
from numba import __version__ as numba_version
from numba import config, cuda, float64, int64
from scipy.sparse import csr_matrix, vstack

from Gimntools.GimnRec.listmode import _trace_3d_segments
from Gimntools.GimnRec.projectors import _siddon_segments, _sub_offset


_THREADS = 128
_trace_plane = cuda.jit(device=True)(_siddon_segments.py_func)
_radial_offset = cuda.jit(device=True)(_sub_offset.py_func)
_trace_volume = cuda.jit(device=True)(_trace_3d_segments.py_func)


@cuda.jit
def _trace_parallel(first, cosines, sines, distances, n, pixel, corner, bin_width, samples,
                    rows, columns, lengths, counts):
    i = cuda.grid(1)
    if i >= len(counts):
        return
    radial, angle = divmod(first + i // samples, len(cosines))
    r = distances[radial, angle] + _radial_offset(i % samples, samples, bin_width)
    c, s = cosines[angle], sines[angle]
    px, py = r * c, r * s
    length = n * pixel
    counts[i] = _trace_plane(px + length*s, py - length*c, px - length*s, py + length*c,
                             n, pixel, corner, rows[i], columns[i], lengths[i])


@cuda.jit
def _trace_lors(rays, shape, spacing, corner, columns, lengths, counts):
    i = cuda.grid(1)
    if i < len(rays):
        index = cuda.local.array(3, int64)
        step = cuda.local.array(3, int64)
        faces = cuda.local.array(3, float64)
        strides = cuda.local.array(3, float64)
        counts[i] = _trace_volume(rays[i], shape, spacing, corner, columns[i], lengths[i],
                                  index, step, faces, strides)


@cuda.jit
def _sparse_dot(data, indices, indptr, values, output):
    entry = cuda.grid(1)
    width = values.shape[1]
    row, col = divmod(entry, width)
    if row < output.shape[0]:
        value = 0.
        for i in range(indptr[row], indptr[row + 1]):
            value += data[i] * values[indices[i], col]
        output[row, col] = value


def _blocks(size):
    return (size + _THREADS - 1) // _THREADS


def _trace_buffers(count, width):
    return (cuda.device_array((count, width), dtype=np.int32),
            cuda.device_array((count, width), dtype=np.float64),
            cuda.device_array(count, dtype=np.int32))


def _pack_rows(columns, coefficients, counts, shape, samples=1, rows=None):
    counts = counts.copy_to_host()
    valid = np.arange(columns.shape[1])[None, :] < counts[:, None]
    ids = columns.copy_to_host()[valid]
    if rows is not None:
        ids += rows.copy_to_host()[valid] * int(np.sqrt(shape[1]))
    weights = coefficients.copy_to_host()[valid] / samples
    per_bin = counts.reshape(-1, samples).sum(axis=1)
    indptr = np.r_[0, np.cumsum(per_bin, dtype=np.int64)]
    return csr_matrix((weights, ids, indptr), shape=shape)


def _use_legacy_context_symbols(driver):
    """Use the driver symbols that match the legacy Numba ctypes prototypes."""
    resolve = driver._find_api

    def find_api(name):
        if name in ("cuCtxGetDevice", "cuCtxSynchronize"):
            return getattr(driver.lib, name)
        return resolve(name)

    driver._find_api = find_api


class CudaReconstruction:
    """Own CUDA buffers for a reconstruction and reuse matrix orientations."""

    def __init__(self):
        if not config.ENABLE_CUDASIM and tuple(map(int, numba_version.split(".")[:2])) < (0, 63):
            from numba.cuda.cudadrv import driver

            # Backport NVIDIA/numba-cuda#185 for the bundled ctypes driver.
            # Newer NVIDIA drivers provide _v2 symbols with a different ABI.
            if not getattr(driver, "USE_NV_BINDING", True):
                _use_legacy_context_symbols(driver.driver)
        if not cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use CPU or install a compatible NVIDIA driver and CUDA runtime")
        free, _ = cuda.current_context().get_memory_info()
        self.cache_budget = min(1024**3, int(free // 3)) if isfinite(free) else 128 * 1024**2
        self.cache = OrderedDict()
        self.cached_bytes = 0

    def _sparse_buffers(self, matrix, transpose):
        key = (id(matrix), transpose)
        cached = self.cache.pop(key, None)
        if cached is not None:
            owner, buffers, size = cached
            if owner() is matrix:
                self.cache[key] = cached
                return buffers
            self.cached_bytes -= size
        csr = (matrix.T if transpose else matrix).tocsr()
        arrays = (csr.data, csr.indices, csr.indptr)
        bytes_used = sum(a.nbytes for a in arrays)
        while self.cache and self.cached_bytes + bytes_used > self.cache_budget:
            _, (_, _, old_size) = self.cache.popitem(last=False)
            self.cached_bytes -= old_size
        buffers = tuple(cuda.to_device(np.ascontiguousarray(a)) for a in arrays)
        if bytes_used <= self.cache_budget:
            self.cache[key] = (ref(matrix), buffers, bytes_used)
            self.cached_bytes += bytes_used
        return buffers

    def sparse_product(self, matrix, vector, *, transpose=False):
        array = np.asarray(vector, dtype=np.float64)
        squeezed = array.ndim == 1
        values = np.ascontiguousarray(array[:, None] if squeezed else array)
        rows = matrix.shape[1] if transpose else matrix.shape[0]
        if not rows or not values.shape[1]:
            result = np.empty((rows, values.shape[1]), dtype=np.float64)
        else:
            output = cuda.device_array((rows, values.shape[1]), dtype=np.float64)
            buffers = self._sparse_buffers(matrix, transpose)
            _sparse_dot[_blocks(output.size), _THREADS](*buffers, cuda.to_device(values), output)
            result = output.copy_to_host()
        return result[:, 0] if squeezed else result

    def trace_parallel(self, angles, distances, n, pixel, corner, bin_width, samples):
        angles = np.asarray(angles, dtype=np.float64)
        if distances.ndim == 1:
            distances = np.broadcast_to(distances[:, None], (len(distances), len(angles)))
        cosines, sines = cuda.to_device(np.cos(angles)), cuda.to_device(np.sin(angles))
        radial = cuda.to_device(np.ascontiguousarray(distances))
        width = 2*n + 2
        bin_count = distances.size
        bins_per_batch = max(1, 2**19 // (width * samples))
        matrices = []
        for first in range(0, bin_count, bins_per_batch):
            bins = min(bins_per_batch, bin_count - first)
            count = bins * samples
            columns, lengths, counts = _trace_buffers(count, width)
            rows = cuda.device_array((count, width), dtype=np.int32)
            _trace_parallel[_blocks(count), _THREADS](first, cosines, sines, radial, n, pixel, corner,
                bin_width, samples, rows, columns, lengths, counts)
            matrices.append(_pack_rows(columns, lengths, counts, (bins, n*n), samples, rows))
        return vstack(matrices, format="csr")

    def trace_lors(self, rays, shape, spacing, corner):
        matrix_shape = (len(rays), int(np.prod(shape)))
        if not len(rays):
            return csr_matrix(matrix_shape, dtype=np.float64)
        columns, lengths, counts = _trace_buffers(len(rays), int(shape.sum()))
        _trace_lors[_blocks(len(rays)), _THREADS](cuda.to_device(np.ascontiguousarray(rays)),
            cuda.to_device(shape), cuda.to_device(spacing), cuda.to_device(corner), columns, lengths, counts)
        return _pack_rows(columns, lengths, counts, matrix_shape)
