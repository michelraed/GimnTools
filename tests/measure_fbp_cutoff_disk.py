"""Resolution/noise trade-off of the cutoff, on a phantom that HAS high
frequencies (a sharp uniform disk), unlike the smooth Gaussian.

Exact parallel projections of a disk of radius R, amplitude A:
    p(r) = 2*A*sqrt(R^2 - r^2)   for |r| < R,  0 otherwise.
Poisson noise is added to a scaled copy so the noise term is realistic.
"""
import numpy as np
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.filters import make_fbp_filter

FOV, NR, NANG, R, AMP = 40.0, 256, 180, 8.0, 1.0
bin_mm = FOV / NR
r = np.linspace(-FOV / 2 + bin_mm / 2, FOV / 2 - bin_mm / 2, NR)
p = np.where(np.abs(r) < R, 2 * AMP * np.sqrt(np.clip(R ** 2 - r ** 2, 0, None)), 0.0)
clean = np.repeat(p[None, :, None], NANG, axis=2)

rng = np.random.default_rng(1)
SCALE = 300.0                       # counts per bin at the disk centre chord
noisy = rng.poisson(clean * SCALE).astype(float) / SCALE
angles = np.linspace(0, 180, NANG, endpoint=False)


def edge_width(img, pix_mm):
    """10-90 % rise distance across the disk edge, along a central row."""
    prof = img[img.shape[0] // 2]
    c = len(prof) // 2
    half = prof[c:]                                  # centre -> outside
    lo, hi = 0.1 * AMP, 0.9 * AMP

    def cross(level):
        idx = np.where(half < level)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        if i == 0:
            return 0.0
        y0, y1 = half[i - 1], half[i]
        return (i - 1 + (y0 - level) / (y0 - y1)) * pix_mm
    a, b = cross(hi), cross(lo)
    return None if a is None or b is None else b - a


for label, sino in (("noiseless", clean), ("Poisson noise", noisy)):
    print(f"\n== {label} ==")
    print(f"{'window':<13}{'cutoff':>7}{'mean in disk':>14}{'/truth':>9}"
          f"{'noise SD in':>13}{'10-90% edge':>13}")
    for window in ("ramp", "hann", "butterworth"):
        for cutoff in (1.0, 0.7, 0.5, 0.3, 0.15):
            rec = siddon_reconstructor(sino, fov_mm=FOV, sino_range_mm=FOV,
                                       verbose=False)
            img = rec.fbp(angles,
                          filter_type=make_fbp_filter(window, cutoff=cutoff,
                                                      order=5))[0]
            n = img.shape[0]
            y, x = np.mgrid[0:n, 0:n]
            cc = (n - 1) / 2
            rr = np.hypot(x - cc, y - cc) * rec.img_pixel
            inside = rr < 0.6 * R
            ew = edge_width(img, rec.img_pixel)
            print(f"{window:<13}{cutoff:>7.2f}{img[inside].mean():>14.4f}"
                  f"{img[inside].mean()/AMP:>9.4f}"
                  f"{img[inside].std():>13.4f}"
                  f"{(f'{ew:.3f}' if ew is not None else 'n/a'):>13}")
