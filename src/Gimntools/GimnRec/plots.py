"""Matplotlib helpers for inspecting sinograms and reconstructions.

All image axes are BIN CENTRES, matching
:func:`Gimntools.GimnRec.quality_metrics.image_axes` and the grid used by
:class:`~Gimntools.GimnRec.reconstructor.siddon_reconstructor`; ``imshow``
extents remain bin EDGES, as matplotlib expects.
"""

import numpy as np
from matplotlib import pyplot as plt

from Gimntools.GimnRec.quality_metrics import fwhm_by_bins, image_axes

__all__ = ["plot_recon_mm", "plot_profiles_fwhm", "plot_comparison_mm",
           "plot_stages_mm"]


def _axis_mm(n, fov):
    """Bin-centre coordinates (mm) of ``n`` pixels spanning ``fov``.

    .. note:: **Fixed in 0.4.0.** These plots previously used
       ``linspace(-fov/2, fov/2, N)``, which places samples on the bin EDGES:
       the sample spacing became ``fov/(N-1)`` instead of ``fov/N``, so every
       distance read off a figure -- FWHM included -- was stretched by
       ``N/(N-1)`` (0.8 % at N=128) and shifted by half a pixel relative to
       the reconstruction grid and to ``quality_metrics``.
    """
    return (np.arange(int(n)) + 0.5) * (float(fov) / int(n)) - float(fov) / 2.0


def _compute_fwhm(mm, vals, baseline="min"):
    """Backward-compatible wrapper over
    :func:`Gimntools.GimnRec.quality_metrics.fwhm_by_bins`.

    .. note:: **Changed in 0.4.0.** This helper used ``max/2`` as the half
       maximum while ``quality_metrics`` used ``(max+min)/2``, so the same
       profile yielded two different FWHM values depending on which module
       reported it. Both now share one implementation and default to the
       background-corrected ``'min'`` convention. Pass ``baseline='zero'`` to
       recover the old plot numbers.
    """
    return fwhm_by_bins(vals, mm, baseline=baseline)


def plot_recon_mm(rec2d, fov, title="", cmap='gray', ax=None):
    ext = [-fov/2, fov/2, -fov/2, fov/2]
    own = ax is None
    if own: fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(rec2d, cmap=cmap, origin='lower', extent=ext)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Counts")
    if own: plt.tight_layout(); plt.show()


def plot_profiles_fwhm(rec2d, fov, title="", baseline="min"):
    """Image plus horizontal/vertical profiles through the maximum, with FWHM.

    ``baseline`` is forwarded to
    :func:`~Gimntools.GimnRec.quality_metrics.fwhm_by_bins`.
    """
    N = rec2d.shape[0]; mm = _axis_mm(N, fov)
    iy, ix = np.unravel_index(np.argmax(rec2d), rec2d.shape)
    ph = rec2d[iy, :]; pv = rec2d[:, ix]
    fh, (lh,rh) = _compute_fwhm(mm, ph, baseline)
    fv, (lv,rv) = _compute_fwhm(mm, pv, baseline)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ext = [-fov/2, fov/2, -fov/2, fov/2]
    axes[0].imshow(rec2d, cmap='gray', origin='lower', extent=ext)
    axes[0].axhline(mm[iy], c='cyan', lw=0.8, ls='--')
    axes[0].axvline(mm[ix], c='lime', lw=0.8, ls='--')
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)"); axes[0].set_title("Image")

    axes[1].plot(mm, ph, 'b-', lw=1.5)
    axes[1].axhline((ph.max()+ph.min())/2 if baseline == 'min' else ph.max()/2,
                    c='r', ls=':', lw=0.8)
    axes[1].axvline(lh, c='r', ls='--', lw=0.8); axes[1].axvline(rh, c='r', ls='--', lw=0.8)
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("Counts")
    axes[1].set_title(f"Horizontal profile — FWHM={fh:.2f}mm"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(mm, pv, 'g-', lw=1.5)
    axes[2].axhline((pv.max()+pv.min())/2 if baseline == 'min' else pv.max()/2,
                    c='r', ls=':', lw=0.8)
    axes[2].axvline(lv, c='r', ls='--', lw=0.8); axes[2].axvline(rv, c='r', ls='--', lw=0.8)
    axes[2].set_xlabel("y (mm)"); axes[2].set_ylabel("Counts")
    axes[2].set_title(f"Vertical profile — FWHM={fv:.2f}mm"); axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13); plt.tight_layout(); plt.show()
    return {'fwhm_h': fh, 'fwhm_v': fv, 'peak_mm': (mm[ix], mm[iy])}


def plot_comparison_mm(osem, mlem, fbp, sl=7, fov=30, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ext = [-fov/2, fov/2, -fov/2, fov/2]
    for ax,rec,lab in zip(axes,[osem,mlem,fbp],["OSEM","MLEM","FBP"]):
        i = min(sl, rec.shape[0]-1)
        im = ax.imshow(rec[i], cmap='gray', origin='lower', extent=ext)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_title(lab)
        plt.colorbar(im, ax=ax, label="Counts")
    fig.suptitle(f"{title} — Slice {sl}", fontsize=13)
    plt.tight_layout(); plt.show()


def plot_stages_mm(stages, fov=30, title="", sl=7):
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1: axes = [axes]
    for ax, (label, s) in zip(axes, stages):
        i = min(sl, s.shape[0]-1)
        ext = [0, 180, -fov/2, fov/2]
        im = ax.imshow(s[i], aspect='auto', cmap='hot', origin='lower', extent=ext)
        ax.set_xlabel("Angle (deg)"); ax.set_ylabel("Distance (mm)"); ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{title} — Slice {sl}", fontsize=13)
    plt.tight_layout(); plt.show()

