"""Tests for the resolution model (psf_kernel) and the inter-iteration
regularization, added in 0.4.0.

Two properties are checked:
  * the radial detector-response operator C is applied as an EXACT adjoint
    pair (C forward / C^T backward) for ANY kernel, not only for symmetric
    odd-length ones;
  * the Gaussian smoothing between iterations is exposed under a name that
    says what it is (regularization_*), with the old gauss_* names still
    accepted.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.filters import (make_psf_kernel, convolve_sino_1d,
                                       convolve_sino_1d_adj)

ok = []
labelled = []


def check(label, cond):
    ok.append(bool(cond))
    labelled.append((label, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")


rng = np.random.default_rng(5)
warnings.filterwarnings("ignore", category=UserWarning)

r = siddon_reconstructor(np.zeros((1, 32, 20)), fov_mm=32.0,
                         sino_range_mm=32.0, n_img=64, verbose=False)
angles = (np.arange(20) + 0.5) * 9.0
ang = np.deg2rad(angles)

# ---- 1. C alone is an exact adjoint pair for every kernel shape -----------
s1 = rng.random((32, 20))
s2 = rng.random((32, 20))
KERNELS = {
    "odd symmetric (default Gaussian)": make_psf_kernel(3.0, r.sino_pixel),
    "odd asymmetric": np.array([.05, .10, .15, .30, .40]),
    "even symmetric": np.array([.25, .25, .25, .25]),
    "even asymmetric": np.array([.1, .2, .3, .4]),
}
for name, k in KERNELS.items():
    k = np.asarray(k, float)
    k = k / k.sum()
    a = float(np.sum(convolve_sino_1d(s1, k) * s2))
    b = float(np.sum(s1 * convolve_sino_1d_adj(s2, k)))
    check(f"C^T exact — {name}", abs(a - b) <= 1e-12 * abs(a))

# ---- 2. full operator (C.A) adjointness, including sub-rays ---------------
x = rng.random((64, 64))
y = rng.random((32, 20))
for name, k in KERNELS.items():
    k = np.asarray(k, float)
    k = k / k.sum()
    a = float(np.sum(r._project(x, ang, k) * y))
    b = float(np.sum(x * r._backproject(y, ang, k)))
    check(f"<C A x, y> == <x, A^T C^T y> — {name}",
          abs(a - b) <= 1e-10 * abs(a))

# ---- 3. the default kernel path is bit-identical to the old code ----------
k = make_psf_kernel(3.0, r.sino_pixel)
old = np.stack([np.convolve(s1[:, t], k, mode="same") for t in range(20)],
               axis=1)
check("forward C bit-identical to the pre-0.4.0 implementation",
      np.array_equal(old, convolve_sino_1d(s1, k)))
check("adjoint C bit-identical for the symmetric odd default",
      np.array_equal(old, convolve_sino_1d_adj(s1, k)))

# ---- 4. kernel longer than the radial axis is rejected, not silently cropped
try:
    convolve_sino_1d(s1, np.ones(64) / 64)
    check("over-long kernel rejected", False)
except ValueError:
    check("over-long kernel rejected", True)

# ---- 5. regularization naming --------------------------------------------
sino = rng.random((1, 32, 20))
rr = siddon_reconstructor(sino, fov_mm=32.0, sino_range_mm=32.0,
                          n_img=64, verbose=False)
v_new = rr.osem(angles, iterations=4, subsets_n=4,
                regularization_every=2, regularization_sigma=0.6,
                verbose=False)
v_old = rr.osem(angles, iterations=4, subsets_n=4,
                gauss_every=2, gauss_sigma=0.6, verbose=False)
check("deprecated gauss_* aliases give identical results",
      np.array_equal(v_new, v_old))

v_off = rr.osem(angles, iterations=4, subsets_n=4,
                regularization_every=0, verbose=False)
check("regularization actually changes the result",
      not np.allclose(v_new, v_off))

try:
    rr.osem(angles, iterations=2, subsets_n=4,
            regularization_every=2, gauss_every=5, verbose=False)
    check("conflicting old/new names rejected", False)
except ValueError:
    check("conflicting old/new names rejected", True)

try:
    rr.osem(angles, iterations=2, subsets_n=4,
            regularization_every=2, regularization_sigma=0.0, verbose=False)
    check("regularization without sigma rejected", False)
except ValueError:
    check("regularization without sigma rejected", True)

check("REGULARIZATION_NOTE reaches the public docstring",
      "REGULARIZATION" in (siddon_reconstructor.osem.__doc__ or "")
      and "NOT part of the system model" in
      (siddon_reconstructor.mlem.__doc__ or ""))

# ---- 6. the two mechanisms remain independent ----------------------------
# psf_kernel must appear in the sensitivity image; regularization must not.
sens_psf = rr._backproject(np.ones((32, 20)), ang, k)
sens_no = rr._backproject(np.ones((32, 20)), ang, None)
check("psf_kernel is part of the sensitivity image (system model)",
      not np.allclose(sens_psf, sens_no))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
def test_all_checks_pass():
    """pytest entry point: every check above must have passed."""
    failed = [lbl for lbl, good in labelled if not good]
    assert not failed, "failed checks: " + ", ".join(failed)


if __name__ == "__main__":
    sys.exit(0 if all(ok) else 1)
