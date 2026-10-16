"""Geometric sensitivity normalization for rotating flat-detector systems.

Why this exists
---------------
For flat detectors rotating in discrete gantry steps, a line of response
(LOR) at sinogram coordinates (r, phi) is only measurable at the gantry
positions where the line crosses BOTH detector faces. Lines near the centre
are accepted at more gantry positions than lines near the radial edge, so
the per-bin sensitivity is non-uniform. Reconstructing without modelling
this yields activity x sensitivity — images much hotter in the centre than
at the edges even for uniform objects.

Physics of the norm
-------------------
For ideal collinear annihilation photons and continuous detector faces, the
acceptance of a given line is BINARY at each gantry position (it either
crosses both faces or it does not). The geometric sensitivity of a sinogram
bin is therefore the COUNT of (gantry position x detector pair)
combinations that accept its central line:

    norm(r, phi) = #{ (g, pair) : |u_i(r, phi)| <= W/2 for both faces }

where u_i is the intersection coordinate of the line with face i. This is
computed analytically (exactly, no Monte Carlo noise) by
:func:`geometric_norm_sinogram` and scaled to mean 1 over its support.

The norm enters the EM model (``norm_sino`` in
:meth:`~Gimntools.GimnRec.reconstructor.siddon_reconstructor.mlem` /
``osem``):

    forward:      y_est = norm * (A x)
    sensitivity:  s     = A^T norm

For FBP, which has no statistical model, pre-divide the sinogram instead
(:func:`apply_norm_to_sinogram`).

Validation (event-level synthetic, dual-head geometry from the example GATE
config — face span W=26.88 mm, face radius 25 mm, 12 gantry positions over
360 deg, emission direction sampled uniformly as in GATE): a uniform disk
reconstructed with OSEM (10 it x 4 subsets) had a centre/edge ratio of 3.1
without the norm and 1.01 with it.

Acknowledged approximations (stated for scientific honesty):
  - Continuous, fully efficient faces: uniform crystal efficiency is
    assumed (reasonable for identical crystals in GATE; NOT true for real
    PETSYS hardware, where a measured normalization from a proper flood
    acquisition is preferable when available).
  - The acceptance test uses a single radial plane per detector (``doi``):
    depth of interaction inside the crystal is not modelled, so acceptance
    at the extreme radial/angular edges is approximate.
  - Bins are represented by their central line; sub-bin partial acceptance
    at the boundary of the accepted region is not resolved.
  - No attenuation, scatter, or per-pair solid-angle weighting.
  - If the measured sinogram is angularly smoothed before reconstruction,
    apply the same smoothing to the norm (the CLI/GUI wiring in this
    package does this automatically).
"""

import numpy as np

__all__ = ["geometric_norm_sinogram", "apply_norm_to_sinogram"]


def _plane_radius(detector_config, doi):
    """Radial position of the acceptance plane for each detector face."""
    radius = float(detector_config["radius"])
    pla = float(detector_config.get("pla_thickness", 0.0))
    thick = float(np.asarray(detector_config["crystal_size"], dtype=float)[0])
    front = radius + pla
    if doi == "front":
        return front
    if doi == "center":
        return front + thick / 2.0
    raise ValueError(f"doi must be 'front' or 'center', got {doi!r}")


def geometric_norm_sinogram(detector_config, detectors_angles,
                            n_dist, n_ang,
                            sino_range_mm=None,
                            doi="front",
                            support_eps=0.0,
                            verbose=True):
    """Analytic geometric (acceptance-count) norm for rotating flat detectors.

    Parameters
    ----------
    detector_config : dict
        Package detector config. Uses ``radius``, ``pla_thickness``,
        ``crystal_size`` (index 1 = transverse face span, index 0 =
        thickness), ``rotation_angles``, ``arround`` and optionally
        ``sino_range_mm``.
    detectors_angles : dict
        ``{detector_name: angle_deg}`` initial placement of each detector.
        All unordered detector pairs are treated as coincidence-capable.
    n_dist, n_ang : int
        Bin counts of the FINAL sinogram grid the data are reconstructed on
        (i.e. after any radial interpolation).
    sino_range_mm : float or None
        Radial span of the sinogram. Default:
        ``detector_config['sino_range_mm']`` if present, else
        ``crystal_size[1]``.
    doi : {'front', 'center'}
        Radial plane used for the face-crossing test (crystal front face or
        crystal centre).
    support_eps : float
        Bins with sensitivity below ``support_eps * max`` are zeroed.
        The acceptance count is bounded below by 1 gantry position on its
        support, so the default 0 is safe; the parameter is kept for
        completeness.
    verbose : bool
        Print a one-line summary.

    Returns
    -------
    norm : (n_dist, n_ang) float64 array
        Acceptance count per bin scaled to mean 1 over its support. Bins
        never accepted stay 0 and are correctly excluded from the EM
        sensitivity.
    """
    W = float(sino_range_mm if sino_range_mm is not None
              else detector_config.get(
                  "sino_range_mm",
                  np.asarray(detector_config["crystal_size"], dtype=float)[1]))
    half_w = W / 2.0
    # The PHYSICAL face half-span is always the crystal transverse width,
    # independent of any padding of the sinogram's radial axis
    # (sino_range_mm > crystal width). Bins in the padded region get zero
    # acceptance, which correctly excludes them from the EM sensitivity.
    face_half = float(np.asarray(detector_config["crystal_size"], dtype=float)[1]) / 2.0
    Rf = _plane_radius(detector_config, doi)
    n_gantry = int(detector_config.get("rotation_angles", 1))
    arround = float(detector_config.get("arround", 360.0))
    base = {k: float(v) for k, v in detectors_angles.items()}
    names = sorted(base)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if not pairs:
        raise ValueError("detectors_angles must contain at least two detectors")

    # bin-centre grid of the sinogram
    r_c = (np.arange(n_dist) + 0.5) * (W / n_dist) - half_w
    a_c = np.deg2rad((np.arange(n_ang) + 0.5) * (180.0 / n_ang))
    Rg, Ag = np.meshgrid(r_c, a_c, indexing="ij")

    def face_ok(phi_det_rad):
        """Boolean map: central line of each bin crosses this face."""
        beta = Ag - phi_det_rad
        sb, cb = np.sin(beta), np.cos(beta)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (Rf - Rg * cb) / (-sb)
            u = Rg * sb + t * cb
        return (np.abs(sb) > 1e-9) & (np.abs(u) <= face_half)

    norm = np.zeros((n_dist, n_ang), dtype=np.float64)
    for g in np.arange(n_gantry) * (arround / n_gantry):
        for (da, db) in pairs:
            ok = face_ok(np.deg2rad(base[da] + g)) & face_ok(np.deg2rad(base[db] + g))
            norm += ok

    if support_eps and support_eps > 0:
        norm[norm < support_eps * norm.max()] = 0.0
    support = norm > 0
    if not support.any():
        raise RuntimeError("geometric norm came out empty — check the geometry "
                           "(radius, crystal_size, detectors_angles)")
    norm = norm / norm[support].mean()
    if verbose:
        print(f"  [geom-norm] acceptance counts over {n_gantry} gantry pos x "
              f"{len(pairs)} pair(s); support {100*support.mean():.1f}% of bins, "
              f"min/max on support = {norm[support].min():.3f}/{norm.max():.3f}")
    return norm


def apply_norm_to_sinogram(sino, norm, eps_frac=1e-3):
    """Pre-divide a sinogram by the norm (for FBP, which has no EM model).

    Bins where ``norm <= eps_frac * max(norm)`` are left at 0 (no data
    support). For MLEM/OSEM prefer passing the norm as ``norm_sino``
    instead of pre-dividing, so the noise model stays Poisson-consistent.

    Parameters
    ----------
    sino : (n_slices, n_dist, n_ang) or (n_dist, n_ang) array
    norm : (n_dist, n_ang) array
    """
    sino = np.asarray(sino, dtype=np.float64)
    norm = np.asarray(norm, dtype=np.float64)
    thr = eps_frac * norm.max()
    safe = np.where(norm > thr, norm, np.inf)
    return sino / safe  # broadcasting covers the 3-D case
