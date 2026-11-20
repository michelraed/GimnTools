"""Siddon ray-driven forward/back projectors (2-D, parallel-beam).

GimnToolsV2 - volume-driven reconstruction update
-------------------------------------------------
The original projectors traced exactly ONE infinitely thin ray per radial
bin. That is adequate only while the image pixel size is >= the radial bin
spacing. As soon as the reconstruction grid is made finer than the detector
sampling (which is precisely what "reconstruct on the volume I want" means),
single-ray tracing leaves pixels that no ray ever crosses: the backprojected
image develops periodic gaps/striping and the EM sensitivity image acquires
near-zero pixels.

The fix implemented here is standard detector-bin sub-sampling: each radial
bin of width ``bin_width`` is traced with ``rays_per_bin`` (K) parallel
sub-rays placed at the mid-points of K equal sub-intervals, and the K line
integrals are averaged (weight 1/K each). This approximates the strip
integral over the bin instead of a single line integral.

Two properties are preserved on purpose:

1. **Exact adjointness.** The forward operator divides by K and the
   backprojector distributes ``value / K`` over the same K sub-rays, so
   ``<A x, y> == <x, A^T y>`` holds to machine precision. MLEM/OSEM require
   this: the sensitivity image must be the adjoint of the operator actually
   used in the forward step, otherwise the fixed point is wrong.

2. **Backward compatibility.** With ``rays_per_bin = 1`` (or
   ``bin_width = 0``) the arithmetic reduces exactly to the previous
   single-ray code, so existing results remain reproducible.
"""

import math
import numpy as np
from numba import prange, njit

@njit
def _siddon_segments(x1, y1, x2, y2, N, pixel_size, grid_min, rows, cols, lens):
    """
    Siddon ray trace — stepping algorithm.
    Fill the supplied buffers and return the number of valid entries.
    """
    eps = 1e-12
    Dx = x2 - x1; Dy = y2 - y1
    d12 = math.sqrt(Dx*Dx + Dy*Dy)
    max_n = 2*N + 2
    if d12 < eps:
        return 0

    # Alpha range (parametric: P = P1 + alpha*(P2-P1))
    if abs(Dx) > eps:
        ax1 = (grid_min - x1) / Dx
        ax2 = (grid_min + N*pixel_size - x1) / Dx
        ax_min, ax_max = (ax1, ax2) if ax1 <= ax2 else (ax2, ax1)
    else:
        ax_min = -1e30; ax_max = 1e30

    if abs(Dy) > eps:
        ay1 = (grid_min - y1) / Dy
        ay2 = (grid_min + N*pixel_size - y1) / Dy
        ay_min, ay_max = (ay1, ay2) if ay1 <= ay2 else (ay2, ay1)
    else:
        ay_min = -1e30; ay_max = 1e30

    entry = ax_min if ax_min >= ay_min else ay_min
    exit_at = ax_max if ax_max <= ay_max else ay_max
    a_min = entry if entry >= 0.0 else 0.0
    a_max = exit_at if exit_at <= 1.0 else 1.0
    if a_min >= a_max - eps:
        return 0

    # Starting pixel
    xs = x1 + (a_min + eps) * Dx
    ys = y1 + (a_min + eps) * Dy
    ix = int((xs - grid_min) / pixel_size)
    iy = int((ys - grid_min) / pixel_size)
    ix = 0 if ix < 0 else (N - 1 if ix >= N else ix)
    iy = 0 if iy < 0 else (N - 1 if iy >= N else iy)

    # Step direction and delta-alpha
    if Dx > eps:
        ix_step = 1; da_x = pixel_size / Dx
        a_next_x = (grid_min + (ix+1)*pixel_size - x1) / Dx
    elif Dx < -eps:
        ix_step = -1; da_x = -pixel_size / Dx
        a_next_x = (grid_min + ix*pixel_size - x1) / Dx
    else:
        ix_step = 0; da_x = 1e30; a_next_x = 1e30

    if Dy > eps:
        iy_step = 1; da_y = pixel_size / Dy
        a_next_y = (grid_min + (iy+1)*pixel_size - y1) / Dy
    elif Dy < -eps:
        iy_step = -1; da_y = -pixel_size / Dy
        a_next_y = (grid_min + iy*pixel_size - y1) / Dy
    else:
        iy_step = 0; da_y = 1e30; a_next_y = 1e30

    # Stepping
    a_cur = a_min; cnt = 0
    for _ in range(max_n):
        if a_cur >= a_max - eps:
            break
        if a_next_x < a_next_y:
            a_end = a_next_x if a_next_x <= a_max else a_max
            length = (a_end - a_cur) * d12
            if 0 <= iy < N and 0 <= ix < N and length > eps:
                rows[cnt]=iy; cols[cnt]=ix; lens[cnt]=length; cnt+=1
            a_cur = a_end; a_next_x += da_x; ix += ix_step
        else:
            a_end = a_next_y if a_next_y <= a_max else a_max
            length = (a_end - a_cur) * d12
            if 0 <= iy < N and 0 <= ix < N and length > eps:
                rows[cnt]=iy; cols[cnt]=ix; lens[cnt]=length; cnt+=1
            a_cur = a_end; a_next_y += da_y; iy += iy_step
    return cnt


@njit
def _siddon_trace(x1, y1, x2, y2, N, pixel_size, grid_min):
    width = 2 * N + 2
    rows = np.empty(width, dtype=np.int32)
    cols = np.empty(width, dtype=np.int32)
    lengths = np.empty(width, dtype=np.float64)
    count = _siddon_segments(x1, y1, x2, y2, N, pixel_size, grid_min, rows, cols, lengths)
    return rows, cols, lengths, count


@njit(inline='always')
def _sub_offset(k, K, bin_width):
    """Radial offset of sub-ray k (0..K-1) inside a bin of width bin_width.

    Sub-rays sit at the centres of K equal sub-intervals:
        offset_k = bin_width * ((k + 0.5)/K - 0.5)
    For K = 1 this is 0.0, i.e. the classic central ray.
    """
    if K <= 1 or bin_width <= 0.0:
        return 0.0
    return bin_width * ((k + 0.5) / K - 0.5)


@njit(parallel=True)
def siddon_forward_2d(image, angles_rad, distances, N, pixel_size, grid_min,
                      bin_width=0.0, rays_per_bin=1):
    """
    Forward project a 2-D image -> 2-D sinogram using Siddon.

    image: (N, N), angles_rad: (Nth,), distances: (Nr,)
    bin_width: physical width (mm) of one radial bin. Only used when
               rays_per_bin > 1.
    rays_per_bin: K sub-rays per radial bin. K = 1 reproduces the original
               single-ray behaviour exactly.

    Returns: sinogram (Nr, Nth)
    """
    Nr = distances.shape[0]; Nth = angles_rad.shape[0]
    sino = np.zeros((Nr, Nth), dtype=np.float64)
    L = N * pixel_size  # ray extent
    K = rays_per_bin if rays_per_bin > 1 else 1
    inv_K = 1.0 / K

    for th in prange(Nth):
        cos_t = np.cos(angles_rad[th]); sin_t = np.sin(angles_rad[th])
        for ri in range(Nr):
            r0 = distances[ri, th] if distances.ndim == 2 else distances[ri]
            acc = 0.0
            for k in range(K):
                r = r0 + _sub_offset(k, K, bin_width)
                # Point on the LOR + direction
                px = r * cos_t; py = r * sin_t
                dx = -sin_t; dy = cos_t
                x1 = px - L*dx; y1 = py - L*dy
                x2 = px + L*dx; y2 = py + L*dy

                rows, cols, lens, cnt = _siddon_trace(x1,y1,x2,y2,N,pixel_size,grid_min)
                val = 0.0
                for j in range(cnt):
                    val += image[rows[j], cols[j]] * lens[j]
                acc += val
            sino[ri, th] = acc * inv_K
    return sino


@njit(parallel=True)
def siddon_backward_2d(sinogram, angles_rad, distances, N, pixel_size, grid_min,
                       bin_width=0.0, rays_per_bin=1, n_chunks=1):
    """
    Backproject a 2-D sinogram -> 2-D image using Siddon. Exact adjoint of
    siddon_forward_2d for the same (bin_width, rays_per_bin).

    sinogram: (Nr, Nth). Returns: image (N, N)

    n_chunks: number of independent accumulation buffers. n_chunks = 1 keeps
        the original strictly sequential angular loop (deterministic
        summation order, identical to previous releases). n_chunks > 1
        splits the angular loop across prange, each chunk writing to its own
        buffer, so there is no data race. The result then differs from the
        sequential one only by floating-point summation order (~1e-15
        relative). Memory cost: n_chunks * N * N * 8 bytes.
    """
    Nr = distances.shape[0]; Nth = angles_rad.shape[0]
    K = rays_per_bin if rays_per_bin > 1 else 1
    inv_K = 1.0 / K
    C = n_chunks if n_chunks > 1 else 1
    if C > Nth:
        C = Nth
    L = N * pixel_size

    buf = np.zeros((C, N, N), dtype=np.float64)

    # Chunk c handles angles c, c+C, c+2C, ... and writes ONLY to buf[c],
    # so prange over chunks is race-free.
    for c in prange(C):
        for th in range(c, Nth, C):
            cos_t = np.cos(angles_rad[th]); sin_t = np.sin(angles_rad[th])
            for ri in range(Nr):
                val = sinogram[ri, th]
                if abs(val) < 1e-15:
                    continue
                contrib = val * inv_K
                r0 = distances[ri, th] if distances.ndim == 2 else distances[ri]
                for k in range(K):
                    r = r0 + _sub_offset(k, K, bin_width)
                    px = r*cos_t; py = r*sin_t
                    dx = -sin_t; dy = cos_t
                    x1 = px - L*dx; y1 = py - L*dy
                    x2 = px + L*dx; y2 = py + L*dy

                    rows, cols, lens, cnt = _siddon_trace(x1,y1,x2,y2,N,pixel_size,grid_min)
                    for j in range(cnt):
                        buf[c, rows[j], cols[j]] += contrib * lens[j]

    if C == 1:
        return buf[0]
    img = np.zeros((N, N), dtype=np.float64)
    for c in range(C):
        img += buf[c]
    return img
