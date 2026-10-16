"""Regression and API tests for the volume-driven reconstruction option."""
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from Gimntools.GimnRec.reconstructor import (siddon_reconstructor,
                                             resolve_image_grid)

ok = []
labelled = []


def check(label, cond):
    ok.append(bool(cond))
    labelled.append((label, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")


# ---- 1. grid resolution algebra ------------------------------------------
check("fov+n_img -> voxel", resolve_image_grid(fov_mm=64.0, n_img=128)
      == (64.0, 128, 0.5))
check("n_img+voxel -> fov", resolve_image_grid(n_img=128, voxel_mm=0.5)
      == (64.0, 128, 0.5))
check("fov+voxel -> n_img", resolve_image_grid(fov_mm=64.0, voxel_mm=0.5)
      == (64.0, 128, 0.5))
check("fov only -> legacy n_dist", resolve_image_grid(fov_mm=64.0, n_dist=32)
      == (64.0, 32, 2.0))
try:
    resolve_image_grid(fov_mm=64.0, n_img=128, voxel_mm=0.25)
    check("inconsistent triple rejected", False)
except ValueError:
    check("inconsistent triple rejected", True)
try:
    resolve_image_grid(n_img=128)
    check("under-determined rejected", False)
except ValueError:
    check("under-determined rejected", True)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    f, n, v = resolve_image_grid(fov_mm=70.0, voxel_mm=0.3)
    check("non-divisible FOV snapped to exact voxel and warned",
          n == 233 and abs(v - 0.3) < 1e-12 and abs(f - 69.9) < 1e-9
          and len(w) == 1)

# ---- 2. legacy call path is untouched ------------------------------------
rng = np.random.default_rng(3)
sino = rng.random((2, 128, 24))
r_leg = siddon_reconstructor(sino, fov_mm=70.0, sino_range_mm=26.88,
                             verbose=False)
check("legacy: n_img == n_dist", r_leg.n_img == 128)
check("legacy: rays_per_bin == 1 (auto)", r_leg.rays_per_bin == 1)
check("legacy: FOV untouched", r_leg.fov_mm == 70.0)

angles = (np.arange(24) + 0.5) * (180.0 / 24)
fbp_new = r_leg.fbp(angles)
r_forced = siddon_reconstructor(sino, fov_mm=70.0, sino_range_mm=26.88,
                                rays_per_bin=1, verbose=False)
check("legacy result reproducible with explicit rays_per_bin=1",
      np.array_equal(fbp_new, r_forced.fbp(angles)))

# ---- 3. volume-driven modes ----------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r_fine = siddon_reconstructor.for_volume(sino, n_img=256, voxel_mm=0.105,
                                             sino_range_mm=26.88, verbose=False)
check("matrix+voxel redefines the FOV", abs(r_fine.fov_mm - 256 * 0.105) < 1e-9)
check("auto rays_per_bin > 1 when voxel < bin", r_fine.rays_per_bin > 1)

# coarser grid than the sinogram — impossible in the legacy path
r_coarse = siddon_reconstructor(sino, fov_mm=26.88, sino_range_mm=26.88,
                                n_img=64, verbose=False)
check("coarser grid allowed", r_coarse.n_img == 64
      and r_coarse.rays_per_bin == 1)

vol = r_coarse.osem(angles, iterations=2, subsets_n=4, verbose=False)
check("OSEM returns the requested matrix", vol.shape == (2, 64, 64))
vol_m = r_coarse.mlem(angles, iterations=2, verbose=False)
check("MLEM returns the requested matrix", vol_m.shape == (2, 64, 64))
check("FBP returns the requested matrix",
      r_coarse.fbp(angles).shape == (2, 64, 64))

# ---- 4. sensitivity image has no holes on a fine grid ---------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r_g = siddon_reconstructor(sino, fov_mm=26.88, sino_range_mm=26.88,
                               n_img=384, rays_per_bin=1, verbose=False)
    r_a = siddon_reconstructor(sino, fov_mm=26.88, sino_range_mm=26.88,
                               n_img=384, verbose=False)
ones = np.ones((128, 24))
s_gap = r_g._backproject(ones, np.deg2rad(angles))
s_ok = r_a._backproject(ones, np.deg2rad(angles))
frac_gap = float(np.mean(s_gap < 1e-9 * s_gap.max()))
frac_ok = float(np.mean(s_ok < 1e-9 * s_ok.max()))
print(f"      zero-sensitivity voxels: K=1 -> {100*frac_gap:.2f}%, "
      f"K=auto({r_a.rays_per_bin}) -> {100*frac_ok:.2f}%")
check("sub-rays remove the sensitivity holes", frac_ok < frac_gap)

# ---- 5. parallel backprojection matches the serial one -------------------
r_par = siddon_reconstructor(sino, fov_mm=26.88, sino_range_mm=26.88,
                             n_img=128, backproject_chunks=8, verbose=False)
r_ser = siddon_reconstructor(sino, fov_mm=26.88, sino_range_mm=26.88,
                             n_img=128, backproject_chunks=1, verbose=False)
a = r_par._backproject(ones, np.deg2rad(angles))
b = r_ser._backproject(ones, np.deg2rad(angles))
check("chunked backprojection == serial (float tol)",
      np.allclose(a, b, rtol=1e-12, atol=1e-12))

# ---- 6. geometry() provenance dict ---------------------------------------
g = r_coarse.geometry()
check("geometry() reports voxel and bin sizes",
      set(["n_img", "voxel_mm", "fov_mm", "sino_bin_mm", "rays_per_bin"])
      <= set(g))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
def test_all_checks_pass():
    """pytest entry point: every check above must have passed."""
    failed = [lbl for lbl, good in labelled if not good]
    assert not failed, "failed checks: " + ", ".join(failed)


if __name__ == "__main__":
    sys.exit(0 if all(ok) else 1)
