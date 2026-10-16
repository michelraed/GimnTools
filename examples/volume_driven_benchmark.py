"""Fair comparison at matched noise level (CRC vs background CV).

Comparing two reconstructions at the SAME iteration number is misleading:
radial interpolation low-pass filters the data and therefore trades
resolution for noise. The honest comparison sweeps the iteration number and
compares contrast recovery at a matched background noise level.
"""
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore", category=UserWarning)
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.filters import interpolate_distance

SRANGE, N_TRUE, N_DIST, N_ANG = 32.0, 256, 32, 60
N_IMG, SUBSETS, COUNTS, N_REAL = 128, 4, 3e5, 3
ITERS = [2, 4, 6, 9, 12, 16, 22, 30]

vox_true = SRANGE / N_TRUE
yy, xx = np.mgrid[0:N_TRUE, 0:N_TRUE]
cx = (xx + 0.5) * vox_true - SRANGE / 2
cy = (yy + 0.5) * vox_true - SRANGE / 2
truth = np.zeros((N_TRUE, N_TRUE))
truth[(cx**2 + cy**2) < 10.0**2] = 1.0
for (sx, sy) in [(-4, 0), (4, 0), (0, 5)]:
    truth[((cx - sx)**2 + (cy - sy)**2) < 1.0**2] = 5.0

angles_deg = (np.arange(N_ANG) + 0.5) * (180.0 / N_ANG)
gen = siddon_reconstructor(np.zeros((1, N_DIST, N_ANG)), fov_mm=SRANGE,
                           sino_range_mm=SRANGE, n_img=N_TRUE,
                           rays_per_bin=12, verbose=False)
clean = gen._project(truth, np.deg2rad(angles_deg))
clean = clean * (COUNTS / clean.sum())

# ROIs on the reconstruction grid
v = SRANGE / N_IMG
gy, gx = np.mgrid[0:N_IMG, 0:N_IMG]
px = (gx + 0.5) * v - SRANGE / 2
py = (gy + 0.5) * v - SRANGE / 2
roi_hot = ((px + 4) ** 2 + py ** 2) < 1.0 ** 2
rr = np.sqrt(px ** 2 + py ** 2)
roi_bg = (rr > 7.0) & (rr < 9.5)


def metrics(img):
    hot = img[roi_hot].mean()
    bg = img[roi_bg]
    crc = (hot / bg.mean() - 1.0) / (5.0 - 1.0)
    return crc, bg.std() / bg.mean()


rng = np.random.default_rng(11)
acc = {"A upsample": {i: [] for i in ITERS}, "C volume": {i: [] for i in ITERS}}
for _ in range(N_REAL):
    noisy = rng.poisson(clean).astype(np.float64)[None, ...]
    up = interpolate_distance(noisy, n_out=N_IMG, kind="cubic")
    recA = siddon_reconstructor(up, fov_mm=SRANGE, sino_range_mm=SRANGE,
                                rays_per_bin=1, verbose=False)
    recC = siddon_reconstructor.for_volume(noisy, n_img=N_IMG,
                                           voxel_mm=v, sino_range_mm=SRANGE,
                                           verbose=False)
    for it in ITERS:
        acc["A upsample"][it].append(metrics(
            recA.osem(angles_deg, iterations=it, subsets_n=SUBSETS,
                      regularization_every=0, verbose=False)[0]))
        acc["C volume"][it].append(metrics(
            recC.osem(angles_deg, iterations=it, subsets_n=SUBSETS,
                      regularization_every=0, verbose=False)[0]))

print(f"OSEM subsets={SUBSETS}, {COUNTS:.0e} counts, {N_REAL} realizations")
print(f"{'iters':>6s} | {'A  CRC':>8s} {'A  bgCV':>8s} | "
      f"{'C  CRC':>8s} {'C  bgCV':>8s}")
for it in ITERS:
    a = np.mean(acc["A upsample"][it], axis=0)
    c = np.mean(acc["C volume"][it], axis=0)
    print(f"{it:6d} | {a[0]:8.3f} {a[1]:8.3f} | {c[0]:8.3f} {c[1]:8.3f}")
