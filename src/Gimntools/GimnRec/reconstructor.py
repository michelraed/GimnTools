from Gimntools.GimnRec.projectors import siddon_forward_2d,siddon_backward_2d
from Gimntools.GimnRec.filters import ramLak, convolve_sino_1d, apply_filter_to_sinogram
from math import ceil
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial.transform import Rotation as ROT
import scipy.fft as fft
import warnings
import numpy as np


# ---------------------------------------------------------------------------
#  Subset construction (OSEM)
# ---------------------------------------------------------------------------
def build_subsets(n_ang, subsets_n, order="interleaved"):
    """Return a list of index-arrays partitioning range(n_ang) into subsets.

    order='interleaved' (default, recommended):
        subset k = indices [k, k+S, k+2S, ...]. Each subset samples angles
        spread across the whole angular range, so its sensitivity image is
        closer to (1/S) of the full sensitivity — the balance assumption
        OSEM relies on. This is the textbook choice.

    order='contiguous' (legacy GimnToolsV2 behaviour, np.array_split):
        subset k = a contiguous angular wedge. Kept for reproducibility.

    NOTE (measured): on a synthetic 24-angle capillary phantom the difference
    between the two orderings was second-order (streak anisotropy ~1.90 vs
    ~1.83 at S=8). Interleaving is preferred on principle and gave slightly
    rounder point sources, but it is NOT a cure for few-view streaking.
    """
    idx = np.arange(n_ang)
    S = int(subsets_n)
    if S < 1:
        raise ValueError("subsets_n must be >= 1")
    if S > n_ang:
        raise ValueError(f"subsets_n={S} exceeds number of angles n_ang={n_ang}")
    if order == "contiguous":
        return [s for s in np.array_split(idx, S) if len(s) > 0]
    elif order == "interleaved":
        return [idx[k::S] for k in range(S)]
    else:
        raise ValueError(
            f"invalid order: {order!r} (use 'interleaved' or 'contiguous')")


def check_subset_count(n_ang, subsets_n, min_angles_per_subset=3):
    """Warn if OSEM is over-subsetted for the available number of angles.

    Rationale (measured, not assumed): with n_ang=24 gantry angles, going to
    subsets_n=24 (one angle per subset) roughly DOUBLED the reconstruction
    error and pushed capillary streak-anisotropy from ~1.85 to ~2.44 in a
    controlled test, because each subset's sensitivity image has large null
    regions and the per-subset update becomes wildly anisotropic. For
    subsets_n between 2 and ~n_ang/3 the metrics were essentially flat.
    So the practical guidance is: keep at least a few angles per subset.
    """
    ang_per_sub = n_ang / max(int(subsets_n), 1)
    if ang_per_sub < min_angles_per_subset:
        warnings.warn(
            f"[OSEM] subsets_n={subsets_n} with only {n_ang} angles gives "
            f"{ang_per_sub:.1f} angle(s)/subset. Over-subsetting causes "
            f"streaking/distortion in few-view data (measured). Recommended: "
            f"subsets_n <= {max(1, n_ang // min_angles_per_subset)} "
            f"(>= {min_angles_per_subset} angles/subset).",
            stacklevel=2,
        )


MAX_AUTO_RAYS_PER_BIN = 16


def resolve_image_grid(fov_mm=None, n_img=None, voxel_mm=None,
                       n_dist=None, tol=1e-6):
    """Resolve the reconstruction grid from any two of (fov, n_img, voxel).

    The three quantities obey ``fov_mm = n_img * voxel_mm``. Supplying two of
    them determines the third; supplying all three is allowed only if they are
    consistent within ``tol`` (relative).

    Legacy fallback: if only ``fov_mm`` is given, ``n_img`` falls back to
    ``n_dist`` (the number of radial bins), which is the historical
    GimnToolsV2 behaviour — the image matrix was implicitly dictated by the
    sinogram sampling.

    Returns
    -------
    (fov_mm, n_img, voxel_mm) : tuple[float, int, float]

    Raises
    ------
    ValueError
        If the specification is under-determined or internally inconsistent.
    """
    have_fov = fov_mm is not None
    have_n = n_img is not None
    have_v = voxel_mm is not None

    if have_n:
        if not np.isfinite(n_img) or int(n_img) != n_img:
            raise ValueError("n_img must be an integer")
        n_img = int(n_img)
        if n_img < 1:
            raise ValueError(f"n_img must be >= 1 (got {n_img})")
    if have_v:
        voxel_mm = float(voxel_mm)
        if not np.isfinite(voxel_mm) or voxel_mm <= 0:
            raise ValueError(f"voxel_mm must be > 0 (got {voxel_mm})")
    if have_fov:
        fov_mm = float(fov_mm)
        if not np.isfinite(fov_mm) or fov_mm <= 0:
            raise ValueError(f"fov_mm must be > 0 (got {fov_mm})")

    if have_fov and have_n and have_v:
        if abs(fov_mm - n_img * voxel_mm) > tol * max(fov_mm, 1.0):
            raise ValueError(
                f"Inconsistent grid specification: fov_mm={fov_mm} but "
                f"n_img*voxel_mm={n_img * voxel_mm}. Supply only two of "
                "(fov_mm, n_img, voxel_mm).")
        return fov_mm, n_img, voxel_mm

    if have_n and have_v:
        return n_img * voxel_mm, n_img, voxel_mm

    if have_fov and have_n:
        return fov_mm, n_img, fov_mm / n_img

    if have_fov and have_v:
        n = int(round(fov_mm / voxel_mm))
        if n < 1:
            raise ValueError(
                f"voxel_mm={voxel_mm} is larger than fov_mm={fov_mm}")
        fov_exact = n * voxel_mm
        if abs(fov_exact - fov_mm) > tol * fov_mm:
            warnings.warn(
                f"[grid] fov_mm={fov_mm:.4f} is not an integer multiple of "
                f"voxel_mm={voxel_mm:.4f}; the FOV was adjusted to "
                f"{fov_exact:.4f} mm ({n} px) so that the voxel size is "
                "exactly as requested.", stacklevel=3)
        return fov_exact, n, voxel_mm

    if have_fov:
        if n_dist is None:
            raise ValueError(
                "Only fov_mm was given and no n_dist is available to fall "
                "back on. Provide n_img or voxel_mm.")
        return fov_mm, int(n_dist), fov_mm / int(n_dist)

    raise ValueError(
        "Under-determined grid: give fov_mm, or two of "
        "(fov_mm, n_img, voxel_mm).")



def _resolve_regularization(regularization_every, regularization_sigma,
                            gauss_every, gauss_sigma,
                            default_every=0, default_sigma=0.5):
    """Resolve the inter-iteration regularization settings.

    ``gauss_every`` / ``gauss_sigma`` are the historical names, kept working
    for backward compatibility. The new names state what the operation
    actually is: an inter-iteration REGULARIZATION of the image estimate,
    not a resolution model.
    """
    if (regularization_every is not None and gauss_every is not None
            and int(regularization_every) != int(gauss_every)):
        raise ValueError(
            f"regularization_every={regularization_every} conflicts with the "
            f"deprecated gauss_every={gauss_every}; pass only one.")
    if (regularization_sigma is not None and gauss_sigma is not None
            and float(regularization_sigma) != float(gauss_sigma)):
        raise ValueError(
            f"regularization_sigma={regularization_sigma} conflicts with the "
            f"deprecated gauss_sigma={gauss_sigma}; pass only one.")

    every = regularization_every
    if every is None:
        every = gauss_every if gauss_every is not None else default_every
    sigma = regularization_sigma
    if sigma is None:
        sigma = gauss_sigma if gauss_sigma is not None else default_sigma

    every = int(every or 0)
    sigma = float(sigma)
    if every > 0 and sigma <= 0:
        raise ValueError(
            "regularization_every > 0 requires regularization_sigma > 0")
    return every, sigma


REGULARIZATION_NOTE = """
    Inter-iteration regularization (regularization_every / regularization_sigma)
    ---------------------------------------------------------------------------
    Every ``regularization_every`` iterations the current image estimate is
    smoothed with an isotropic 2-D Gaussian of standard deviation
    ``regularization_sigma`` (in PIXELS, so its physical width scales with
    the voxel size — re-tune it whenever the grid changes).

    This is REGULARIZATION, not a resolution model. Three consequences that
    must be stated when reporting results:

      * it is NOT part of the system model ``H``; unlike ``psf_kernel`` it
        does not appear in the sensitivity image, so forward and adjoint
        remain consistent, but
      * it breaks the monotone likelihood ascent of EM: the iterates no
        longer maximize the Poisson log-likelihood, and
      * it changes the fixed point of the algorithm. The result depends
        jointly on (iterations, subsets, period, sigma), so all four must be
        reported together; an image obtained with smoothing is not
        comparable with one obtained without it at the same iteration count.

    ``regularization_every = 0`` disables it. For a principled alternative
    use ``relaxation < 1`` (which damps noise WITHOUT changing the fixed
    point) or a penalized-likelihood/MAP formulation.

    PARAMETER WARNING (measured): a large sigma DESTROYS fine structure. On
    a capillary phantom, sigma = 2.0 (every 5 iterations) increased the
    error ~14x and melted the capillaries into round blobs. Use sigma <= ~0.5
    for capillaries; 2.0 is aggressive filtering, not resolution recovery.
"""


class siddon_reconstructor:
    """
    Reconstructor based on a 2-D Siddon projector/backprojector.

    Geometry parameters
    -------------------
    fov_mm:        PHYSICAL size of the reconstructed image (mm).
                   The image covers [-fov/2, +fov/2] in x and y.
    sino_range_mm: range of the sinogram distances (mm).
                   The sinogram r axis covers [-sino_range/2, +sino_range/2].
                   If None, fov_mm is used.
    n_img:         reconstructed image size in pixels (N x N).
    voxel_mm:      in-plane voxel size (mm/px) of the reconstructed image.

    ``fov_mm``, ``n_img`` and ``voxel_mm`` are linked by
    ``fov_mm = n_img * voxel_mm``; supply any two. Supplying only ``fov_mm``
    keeps the legacy behaviour ``n_img = n_dist`` (image matrix dictated by
    the sinogram sampling).

    VOLUME-DRIVEN MODE (recommended)
    --------------------------------
    Historically the only way to change the reconstructed matrix in this
    package was to radially UPSAMPLE the sinogram (``interpolate_distance``)
    and let ``n_img = n_dist`` follow. That is undesirable for two reasons:

      * cubic interpolation of Poisson counts invents no information, but it
        DOES correlate neighbouring bins and can produce negative values,
        which violates the independent-Poisson likelihood MLEM/OSEM assume;
      * the image grid ends up as a side effect of a pre-processing choice
        instead of being an explicit reconstruction parameter.

    Specifying ``n_img`` / ``voxel_mm`` directly reconstructs on the grid you
    want from the ORIGINAL measured bins. Note this does not add information:
    a finer grid cannot exceed the resolution supported by the radial/angular
    sampling; it only removes the resampling step and makes the voxel size an
    explicit, reportable parameter.

    Centre of rotation
    ------------------
    cor_mm:       radial position (mm) of the rotation axis in the sinogram.
                  A misaligned axis appears in a parallel-beam sinogram as a
                  CONSTANT radial offset of every angular column (unlike a
                  translated OBJECT, which produces the angle-dependent
                  sinusoid dx*cos(theta) + dy*sin(theta) handled by
                  corrections.apply_cor). Default 0.0 = axis at the centre
                  of the radial axis.

                  The offset is applied to the projector GEOMETRY — the
                  radial coordinate of every bin becomes
                  ``bin_centre - cor_mm`` — and not by shifting the
                  sinogram. Consequences: no interpolation of the counts
                  (so the Poisson statistics MLEM/OSEM assume are
                  preserved), sub-bin offsets are exact rather than
                  resampled, and nothing is lost off the edge of the array.

                  Sign convention: if a centred point source appears at
                  r = +2 mm in the sinogram, set cor_mm = +2.0.

    Detector-bin sub-sampling
    -------------------------
    rays_per_bin: number of parallel sub-rays traced per radial bin
                  ('auto' by default). A single ray per bin is an
                  infinitely thin line: whenever the voxel is SMALLER than
                  the radial bin spacing, voxels between consecutive rays are
                  never crossed and the backprojection/sensitivity image
                  develops periodic gaps. 'auto' selects
                  ``ceil(sino_pixel / img_pixel)`` (capped at
                  MAX_AUTO_RAYS_PER_BIN), i.e. the smallest K that keeps the
                  sub-ray spacing <= one voxel. K = 1 reproduces the previous
                  releases exactly. Cost scales linearly with K.

    backproject_chunks: >1 parallelizes the backprojection over angle chunks
                  (each chunk owns its accumulator, so there is no race).
                  Changes only the floating-point summation order.

    FOV != detector size!
      FOV = reconstructed volume (where the rays are accumulated)
      Detector = physical hardware (determines the r range of the sinogram)

    WARNING (geometry): if fov_mm >> sino_range_mm, the image covers a much
    larger area than the region informed by the data (|r| <= sino_range/2).
    Objects outside that radius are not representable and the effective
    resolution at the centre drops (fewer useful pixels). For central
    capillaries, consider a fov_mm close to sino_range_mm so resolution is
    not wasted.
    """
    def __init__(self, sinogram_3d, fov_mm=None, sino_range_mm=None,
                 n_img=None, voxel_mm=None, rays_per_bin="auto",
                 backproject_chunks=1, cor_mm=0.0, verbose=True,
                 dx_cor_mm=0.0, dy_cor_mm=0.0):
        data = np.asarray(sinogram_3d, dtype=np.float64)
        if data.ndim != 3 or any(d < 1 for d in data.shape):
            raise ValueError("sinogram_3d must be nonempty (slices, radial, angle)")
        if not np.all(np.isfinite(data)):
            raise ValueError("sinogram_3d contains NaN or infinity")
        self.sinogram = data.copy()
        self.n_slices, self.n_dist, self.n_ang = data.shape

        # --- image grid: any two of (fov, n_img, voxel) ---
        self.fov_mm, self.n_img, self.img_pixel = resolve_image_grid(
            fov_mm=fov_mm, n_img=n_img, voxel_mm=voxel_mm, n_dist=self.n_dist)
        self.voxel_mm = self.img_pixel                  # explicit alias (mm/px)
        self.img_min = -self.fov_mm / 2.0               # lower-left corner

        # --- Range of the sinogram distances (may differ from the FOV) ---
        self.sino_range = (float(sino_range_mm) if sino_range_mm is not None
                           else self.fov_mm)
        sino_half = self.sino_range / 2.0
        if not np.isfinite(self.sino_range) or self.sino_range <= 0:
            raise ValueError("sino_range_mm must be positive and finite")
        self.sino_pixel = self.sino_range / self.n_dist  # mm/bin along r
        bin_centres = np.linspace(-sino_half + self.sino_pixel/2,
                                   sino_half - self.sino_pixel/2,
                                   self.n_dist)

        # --- centre-of-rotation offset (constant along theta) -------------
        # The rotation axis is assumed to sit at r = cor_mm in the sinogram
        # instead of r = 0. The correction is applied to the GEOMETRY (the
        # radial coordinate attached to each bin) and NOT by shifting the
        # sinogram: no interpolation, no resampling of the counts, and any
        # fractional offset is handled exactly.
        self.cor_mm = float(cor_mm)
        self.dx_cor_mm = float(dx_cor_mm)
        self.dy_cor_mm = float(dy_cor_mm)
        if not np.all(np.isfinite([self.cor_mm, self.dx_cor_mm, self.dy_cor_mm])):
            raise ValueError("COR offsets must be finite")
        self.distances = bin_centres - self.cor_mm

        # --- detector-bin sub-sampling ---
        self.oversampling = self.sino_pixel / self.img_pixel
        if rays_per_bin in (None, "auto"):
            k = int(ceil(self.oversampling - 1e-9))
            self.rays_per_bin = max(1, min(k, MAX_AUTO_RAYS_PER_BIN))
            if k > MAX_AUTO_RAYS_PER_BIN:
                warnings.warn(
                    f"[grid] auto rays_per_bin wanted {k} but was capped at "
                    f"{MAX_AUTO_RAYS_PER_BIN}. The voxel is {self.oversampling:.1f}x "
                    "finer than the radial bin; residual gaps between sub-rays "
                    "are possible. Consider a larger voxel_mm.", stacklevel=2)
        else:
            try:
                rays_n = int(rays_per_bin)
                if (rays_n < 1 or not np.isfinite(float(rays_per_bin))
                        or float(rays_per_bin) != rays_n):
                    raise ValueError()
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("rays_per_bin must be a positive integer or 'auto'") from exc
            self.rays_per_bin = rays_n
        try:
            chunks_n = int(backproject_chunks)
            if (chunks_n < 1 or not np.isfinite(float(backproject_chunks))
                    or float(backproject_chunks) != chunks_n):
                raise ValueError()
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("backproject_chunks must be a positive integer") from exc
        self.backproject_chunks = chunks_n

        if self.oversampling > 2.0:
            warnings.warn(
                f"[grid] the image grid is {self.oversampling:.1f}x finer than the "
                f"radial sampling (voxel {self.img_pixel:.3f} mm vs bin "
                f"{self.sino_pixel:.3f} mm). Sub-ray sampling keeps the "
                "projector consistent, but a finer grid adds NO information: "
                "the system becomes more under-determined and the iterative "
                "solution noisier. Report the voxel size together with the "
                "radial bin size.", stacklevel=2)

        if verbose:
            print(f"  Siddon: image {self.n_img}x{self.n_img} px, "
                  f"FOV={self.fov_mm:.1f}mm ({self.img_pixel:.3f}mm/px), "
                  f"sino_range={self.sino_range:.1f}mm ({self.n_dist} bins, "
                  f"{self.sino_pixel:.3f}mm/bin), "
                  f"rays/bin={self.rays_per_bin}"
                  + (f", COR={self.cor_mm:+.4f}mm "
                     f"({self.cor_mm/self.sino_pixel:+.3f} bins)"
                     if abs(self.cor_mm) > 0 else ""))

    # -- convenience constructor: "reconstruct THIS volume" ------------------
    @classmethod
    def for_volume(cls, sinogram_3d, n_img=None, voxel_mm=None,
                   sino_range_mm=None, fov_mm=None, **kwargs):
        """Volume-driven constructor.

        Explicitly states the reconstruction grid instead of inheriting it
        from the sinogram sampling. Give any two of (n_img, voxel_mm, fov_mm).

        >>> rec = siddon_reconstructor.for_volume(sino, n_img=256,
        ...                                        voxel_mm=0.25,
        ...                                        sino_range_mm=26.88)
        """
        if sum(v is not None for v in (n_img, voxel_mm, fov_mm)) < 2:
            raise ValueError(
                "for_volume() needs at least two of (n_img, voxel_mm, fov_mm); "
                "otherwise the volume is not defined. Use the normal "
                "constructor for the legacy sinogram-driven behaviour.")
        return cls(sinogram_3d, fov_mm=fov_mm, sino_range_mm=sino_range_mm,
                   n_img=n_img, voxel_mm=voxel_mm, **kwargs)

    def geometry(self):
        """Return the resolved geometry as a dict (for logging/provenance)."""
        return {
            "n_img": self.n_img,
            "voxel_mm": self.img_pixel,
            "fov_mm": self.fov_mm,
            "n_dist": self.n_dist,
            "n_ang": self.n_ang,
            "sino_range_mm": self.sino_range,
            "sino_bin_mm": self.sino_pixel,
            "rays_per_bin": self.rays_per_bin,
            "grid_oversampling": self.oversampling,
            "cor_mm": self.cor_mm,
            "cor_bins": self.cor_mm / self.sino_pixel,
            "dx_cor_mm": self.dx_cor_mm,
            "dy_cor_mm": self.dy_cor_mm,
        }

    def _distances_for_angles(self, angles_rad):
        """Measured-bin coordinates corrected in the *system geometry*.

        If an observed point is shifted by dx*cos(theta)+dy*sin(theta)
        in r, its true detector coordinate is r minus that displacement.
        """
        angles_rad = np.asarray(angles_rad, dtype=np.float64)
        if self.dx_cor_mm == 0.0 and self.dy_cor_mm == 0.0:
            return self.distances
        displacement = (self.dx_cor_mm * np.cos(angles_rad) +
                        self.dy_cor_mm * np.sin(angles_rad))
        return np.ascontiguousarray(self.distances[:, None] - displacement[None, :])

    def _project(self, img, angles_rad, psf_kernel=None):
        """Forward operator  y = C . A . x.

        ``A`` = Siddon line/strip integrals, ``C`` = radial detector-response
        convolution in the sinogram domain (``psf_kernel``). ``C`` models the
        finite detector/positioning resolution and is part of the SYSTEM
        MODEL — it must not be confused with the inter-iteration Gaussian
        regularization applied to the image estimate (see ``mlem``/``osem``).
        """
        sino = siddon_forward_2d(np.ascontiguousarray(img), angles_rad,
                                  self._distances_for_angles(angles_rad), self.n_img,
                                  self.img_pixel, self.img_min,
                                  self.sino_pixel, self.rays_per_bin)
        if psf_kernel is not None:
            sino = convolve_sino_1d(sino, psf_kernel, adjoint=False)
        return sino

    def _backproject(self, sino, angles_rad, psf_kernel=None):
        """Adjoint operator  x = A^T . C^T . y.

        ``C^T`` is applied with ``adjoint=True``, i.e. the true transpose of
        the convolution used in ``_project`` (reversed kernel and mirrored
        crop offset), so that ``<A_full x, y> == <x, A_full^T y>`` holds for
        ANY kernel — not only for symmetric odd-length ones. Before 0.4.0 the
        forward convolution was reused here, which was exact only by
        coincidence for the default Gaussian kernel.
        """
        if psf_kernel is not None:
            sino = convolve_sino_1d(sino, psf_kernel, adjoint=True)
        return siddon_backward_2d(np.ascontiguousarray(sino), angles_rad,
                                   self._distances_for_angles(angles_rad), self.n_img,
                                   self.img_pixel, self.img_min,
                                   self.sino_pixel, self.rays_per_bin,
                                   self.backproject_chunks)

    def fbp(self, angles_deg, filter_type=ramLak, normalize=True,
            pad_factor=2, pad_mode="pow2", legacy_scale=False):
        """Filtered back-projection, slice by slice.

        Parameters
        ----------
        angles_deg : (n_ang,) array
            Projection angles in degrees. Assumed to sample ``[0, 180)``
            uniformly; the angular integration step is taken as
            ``pi / n_ang``.
        filter_type : callable
            Filter factory, see :mod:`Gimntools.GimnRec.filters`.
        normalize : bool
            Apply the FBP quadrature constant (default). See below.
        pad_factor : int
            Zero-padding of the radial FFT, passed to
            :func:`~Gimntools.GimnRec.filters.apply_filter_to_sinogram`.
        pad_mode : {'none', 'pow2', 'exact'}
            Zero-padding PROCEDURE, see
            :func:`~Gimntools.GimnRec.filters.resolve_pad_length`. The
            default reproduces the 0.4.0 behaviour exactly.

        legacy_scale : bool
            Use the pre-0.4.1 quadrature constant. See below; only needed to
            regenerate figures produced with a volume-driven grid before the
            correction.

        Normalization
        -------------
        The back-projector used here is the exact adjoint ``A^T`` of the
        Siddon LINE-INTEGRAL forward projector, so it weights each
        contribution by the intersection LENGTH, whereas the FBP formula
        needs a plain value back-projection. For a family of parallel rays
        spaced ``bin_mm`` apart, the total intersection length inside one
        voxel is ``voxel_mm**2 / bin_mm``; the filters return ``|f|`` in
        cycles per bin, i.e. a factor ``bin_mm`` above the physical ramp, so
        the two ``bin_mm`` factors CANCEL. With the angular step
        ``d(theta) = pi / n_ang`` the constant is::

            f = A^T[ filtered sinogram ] * (pi / n_ang) / voxel_mm**2

        With this constant the reconstruction is in the same physical units
        as the object that produced the projections, and — importantly — the
        value does not depend on how the image grid was chosen.

        .. warning:: **Corrected in 0.4.1.** The constant was previously
           ``(pi / n_ang) / (voxel_mm * bin_mm)``, which cancels only one of
           the two ``bin_mm`` factors and therefore scaled the whole image by
           ``voxel_mm / bin_mm``. That ratio is exactly 1 in the legacy
           sinogram-driven grid (where ``n_img`` follows ``n_dist`` and the
           FOV equals the sinogram range), which is why the error was
           invisible: every previously verified case sat on that diagonal.
           It is NOT 1 as soon as the grid is volume-driven
           (``n_img``/``voxel_mm`` given explicitly) or the radial axis is
           resampled. Measured on the analytic Gaussian blob (exact
           projections, FOV 40 mm, 256 radial bins, 180 views), reconstructed
           peak / truth:

           ======  =========  ==============  =============
           n_img   voxel/bin  old constant    new constant
           ======  =========  ==============  =============
           64      4.000      3.771           0.943
           128     2.000      1.967           0.984
           256     1.000      1.003           1.003
           384     0.667      0.703           1.054
           512     0.500      0.515           1.031
           ======  =========  ==============  =============

           The old column tracks ``voxel/bin``; the new one is flat to within
           the discretisation error of the blob itself (a 3.53 mm FWHM object
           sampled in 0.625 mm voxels loses ~6 % of its peak to partial
           volume, which is the 0.943 entry). MLEM on the same data gives
           0.968-1.002 over the same sweep, i.e. it was already correct, so
           before this fix FBP and MLEM were NOT comparable on a
           volume-driven grid. Reconstructions made with the legacy
           sinogram-driven grid are unaffected and need no regeneration.

        Verified on an analytic Gaussian blob with exact projections
        (reconstructed value / truth at the centre): 0.9891 without radial
        zero-padding, 0.9971 at ``pad_factor=2``, 0.9990 at ``pad_factor=4``;
        stable across 128-512 pixels and 90-720 views.

        ``normalize=False`` returns the unscaled back-projection, reproducing
        the pre-0.4.0 output, which was in arbitrary units and therefore not
        comparable with MLEM/OSEM or between acquisitions.

        Returns
        -------
        (n_slices, n_img, n_img) array
        """
        angles_rad = np.deg2rad(angles_deg)
        rec = np.zeros((self.n_slices, self.n_img, self.n_img))
        for z in range(self.n_slices):
            filtered = apply_filter_to_sinogram(filter_type, self.sinogram[z],
                                                pad_factor=pad_factor,
                                                pad_mode=pad_mode)
            rec[z] = self._backproject(filtered, angles_rad)
        if normalize:
            rec *= (np.pi / len(angles_rad)) / (self.img_pixel *
                                                (self.sino_pixel if legacy_scale
                                                 else self.img_pixel))
        return rec

    def _validate_em_inputs(self, angles_deg, norm_sino, background_sino,
                            iterations, relaxation, ratio_cap):
        if not np.isfinite(self.sinogram).all() or np.any(self.sinogram < 0):
            raise ValueError("MLEM/OSEM require nonnegative measured counts; "
                             "use additive background_sino, not background subtraction")
        angles = np.asarray(angles_deg, dtype=float)
        if angles.shape != (self.n_ang,) or not np.isfinite(angles).all():
            raise ValueError("angles_deg must be finite, with one angle per projection")
        if not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not (np.isfinite(relaxation) and 0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must belong to (0,1]")
        if ratio_cap is not None and (not np.isfinite(ratio_cap) or ratio_cap <= 0):
            raise ValueError("ratio_cap must be positive finite, or None")
        if norm_sino is None:
            norm = np.ones((self.n_dist, self.n_ang), dtype=np.float64)
        else:
            norm = np.asarray(norm_sino, dtype=np.float64)
            if norm.shape != (self.n_dist, self.n_ang) or not np.isfinite(norm).all() or np.any(norm < 0):
                raise ValueError("norm_sino must be nonnegative finite (radial, angle)")
        if background_sino is None:
            background = np.zeros_like(self.sinogram)
        else:
            background = np.asarray(background_sino, dtype=np.float64)
            if background.shape == (self.n_dist, self.n_ang):
                background = np.broadcast_to(background, self.sinogram.shape)
            if (background.shape != self.sinogram.shape or
                    not np.isfinite(background).all() or np.any(background < 0)):
                raise ValueError("background_sino must be nonnegative finite "
                                 "(radial, angle) or (slice, radial, angle)")
        return norm, background

    def mlem(self, angles_deg, iterations=80, psf_kernel=None,
             norm_sino=None,
             regularization_every=None, regularization_sigma=None,
             verbose=True, ratio_cap=1e6, relaxation=1.0, legacy=False,
             gauss_every=None, gauss_sigma=None, background_sino=None):
        """
        MLEM with Siddon + radial detector response + (flood) normalization.

        Two distinct smoothing mechanisms exist and must not be conflated:

          psf_kernel  -> RESOLUTION MODEL. A 1-D radial kernel convolved with
                         the sinogram inside the system model
                         ``H = diag(norm) . C . A``. It appears in BOTH the
                         forward step and the sensitivity image (exact
                         transpose), so it is a statement about the physics
                         of the detector, and the EM fixed point remains the
                         maximum-likelihood solution FOR THAT MODEL.
          regularization_* -> REGULARIZATION. An ad-hoc Gaussian smoothing of
                         the IMAGE ESTIMATE between iterations. See
                         REGULARIZATION_NOTE below.

        (``gauss_every`` / ``gauss_sigma`` are deprecated aliases of
        ``regularization_every`` / ``regularization_sigma``.)

        norm_sino: (n_dist, n_ang) — flood normalization factors.
                   Corrects detector efficiency INSIDE the EM model:
                     Forward:      y_est = norm . C . A . x
                     Sensitivity:  s = A^T . C^T . norm
                   If None, uniform efficiency is assumed (norm = 1).

        ratio_cap: upper limit applied to the ratio y/y_est by CLIPPING
                   (not resetting). Prevents numerical blow-ups without
                   cancelling the direction of the correction. Use None to
                   disable the cap.

        relaxation: under-relaxation factor lambda in (0, 1] applied to the
                   multiplicative update: x <- x . (1 + lambda*(update - 1)).
                   lambda = 1 is standard MLEM. lambda < 1 damps noise
                   (useful at low counts) at the cost of slower convergence.
                   It does not change the fixed point.

        legacy:    if True, reproduces the OLD behaviour (additive 0.3*max
                   term in the denominator and ratio reset to 1.0 above
                   1000) for side-by-side comparison.

        """
        norm_valid, background = self._validate_em_inputs(
            angles_deg, norm_sino, background_sino, iterations, relaxation, ratio_cap)
        reg_every, reg_sigma = _resolve_regularization(
            regularization_every, regularization_sigma,
            gauss_every, gauss_sigma,
            default_every=0, default_sigma=0.5)   # MLEM historical defaults
        angles_rad = np.deg2rad(angles_deg)
        N = self.n_img
        rec = np.zeros((self.n_slices, N, N))

        norm = norm_valid

        sens = self._backproject(norm, angles_rad, psf_kernel)
        sens[sens < 1e-9] = 1e-9

        for z in range(self.n_slices):
            y = self.sinogram[z]
            b = background[z]
            x = np.ones((N, N), dtype=np.float64)
            for it in range(iterations):
                Ax = self._project(np.clip(x, 0, None), angles_rad, psf_kernel)
                if legacy:
                    Ax_norm = Ax * norm + Ax.max() * 0.3 + b
                    Ax_norm[Ax_norm < 1e-10] = 1e-10
                    ratio = np.nan_to_num(y / Ax_norm, nan=1.0)
                    ratio[ratio > 1000] = 1.0
                else:
                    Ax_norm = Ax * norm + b
                    Ax_norm[Ax_norm < 1e-10] = 1e-10
                    ratio = np.nan_to_num(y / Ax_norm, nan=0.0,
                                          posinf=0.0, neginf=0.0)
                    if ratio_cap is not None:
                        np.clip(ratio, 0.0, ratio_cap, out=ratio)

                correction = self._backproject(ratio * norm, angles_rad, psf_kernel)
                update = correction / sens
                if not legacy and relaxation != 1.0:
                    update = 1.0 + relaxation * (update - 1.0)
                x *= update
                x = np.clip(x, 0, None)

                # --- inter-iteration REGULARIZATION (not the system model) ---
                if reg_every > 0 and (it+1) % reg_every == 0:
                    x = gaussian_filter(x, sigma=reg_sigma, mode='nearest')
            rec[z] = x
            if verbose:
                reg_txt = (f", reg sigma={reg_sigma} every {reg_every}it"
                           if reg_every > 0 else ", no regularization")
                print(f"  [MLEM] slice {z} done ({iterations}it{reg_txt})")
        return rec

    def osem(self, angles_deg, iterations=10, subsets_n=8, psf_kernel=None,
             norm_sino=None,
             regularization_every=None, regularization_sigma=None,
             verbose=True, ratio_cap=1e6, relaxation=1.0,
             subset_order="interleaved", warn_subsets=True, legacy=False,
             gauss_every=None, gauss_sigma=None, background_sino=None):
        """
        OSEM with Siddon + radial detector response + (flood) normalization.

        psf_kernel is the RESOLUTION MODEL (inside H, mirrored in the
        sensitivity image); regularization_* is inter-iteration
        REGULARIZATION of the image estimate. See mlem() and
        REGULARIZATION_NOTE for why the distinction matters when reporting.
        ``gauss_every`` / ``gauss_sigma`` remain accepted as deprecated
        aliases.

        norm_sino: same logic as in mlem — enters the forward model.

        ratio_cap / legacy: see the mlem() docstring.

        subset_order : 'interleaved' (default) or 'contiguous'.
            'interleaved' — each subset samples angles spread over the whole
            angular range (default choice; more balanced per-subset
            sensitivity). 'contiguous' reproduces the old behaviour
            (np.array_split), where each subset is a narrow angular wedge.
            Measured effect: small (streak anisotropy ~1.83 vs ~1.90 at
            S=8). It is not a cure for few-view data, but it is the more
            correct choice.

        relaxation : under-relaxation lambda in (0, 1] (see mlem). lambda < 1
            reduces the noise/limit-cycle typical of OSEM at low counts.

        warn_subsets : warn when there are too few angles per subset.

        FIX NOTE (GimnToolsV2): the old version did `coef[coef>100] = 1.0`,
        a reset that erased legitimate structure whenever the projection was
        still ~0 in a region that had counts. That was replaced by a clip
        (ratio_cap). In addition:

        MEASURED CAUSE OF CAPILLARY DISTORTION (not assumed): the dominant
        distortion in few-angle acquisitions (12-24) does NOT come from the
        clamps or the subset ordering. It comes from (1) OVER-SUBSETTING —
        using subsets_n close to n_angles (e.g. 24 subsets for 24 angles)
        doubled the error and increased the streaking; and (2) EXCESSIVE
        REGULARIZATION (sigma ~ 2.0), which melts the capillaries together.
        A residual streak remains in ALL configurations — it is the
        intrinsic sampling limit (24 projections over 180 degrees) and is
        not removable by tuning EM. See the diagnostic notebook.
        """
        norm_valid, background = self._validate_em_inputs(
            angles_deg, norm_sino, background_sino, iterations, relaxation, ratio_cap)
        reg_every, reg_sigma = _resolve_regularization(
            regularization_every, regularization_sigma,
            gauss_every, gauss_sigma,
            default_every=5, default_sigma=0.5)   # OSEM historical defaults
        angles_rad = np.deg2rad(angles_deg)
        n_ang_total = len(angles_rad)
        if warn_subsets and not legacy:
            check_subset_count(n_ang_total, subsets_n)

        N = self.n_img
        rec = np.zeros((self.n_slices, N, N))

        if legacy:
            # Reproduces EXACTLY the old (contiguous) partitioning.
            subset_idx = [s for s in np.array_split(np.arange(n_ang_total), subsets_n)
                          if len(s) > 0]
        else:
            subset_idx = build_subsets(n_ang_total, subsets_n, order=subset_order)

        angles_sub = [angles_rad[s] for s in subset_idx]

        norm_subs = [norm_valid[:, s] for s in subset_idx]

        sens_sub = []
        for asub, nsub in zip(angles_sub, norm_subs):
            s = self._backproject(nsub, asub, psf_kernel)
            s[s < 1e-9] = 1e-9
            sens_sub.append(s)

        for z in range(self.n_slices):
            sino_z = self.sinogram[z]
            sino_subs = [sino_z[:, s] for s in subset_idx]
            bg_subs = [background[z][:, s] for s in subset_idx]
            x = np.ones((N, N), dtype=np.float64)
            for it in range(iterations):
                for i, (sub_sino, asub) in enumerate(zip(sino_subs, angles_sub)):
                    Ax = self._project(np.clip(x, 0, None), asub, psf_kernel)
                    Ax_norm = Ax * norm_subs[i] + bg_subs[i]
                    Ax_norm[Ax_norm < 1e-10] = 1e-10

                    if legacy:
                        coef = np.nan_to_num(sub_sino / Ax_norm, nan=1.0)
                        coef[coef > 100] = 1.0
                    else:
                        coef = np.nan_to_num(sub_sino / Ax_norm, nan=0.0,
                                             posinf=0.0, neginf=0.0)
                        if ratio_cap is not None:
                            np.clip(coef, 0.0, ratio_cap, out=coef)

                    correction = self._backproject(coef * norm_subs[i], asub, psf_kernel)
                    update = correction / sens_sub[i]
                    if not legacy and relaxation != 1.0:
                        update = 1.0 + relaxation * (update - 1.0)
                    x *= update
                    x = np.clip(x, 0, None)

                # --- inter-iteration REGULARIZATION (not the system model) ---
                if reg_every > 0 and (it+1) % reg_every == 0:
                    x = gaussian_filter(x, sigma=reg_sigma, mode='nearest')
            rec[z] = x
            if verbose:
                reg_txt = (f", reg sigma={reg_sigma} every {reg_every}it"
                           if reg_every > 0 else ", no regularization")
                print(f"  [OSEM] slice {z} done ({iterations}it x {len(subset_idx)}sub, "
                      f"order={'legacy/contiguous' if legacy else subset_order}"
                      f"{reg_txt})")
        return rec


# The regularization note is appended programmatically so that the
# docstrings stay plain literals (help() / Sphinx pick them up correctly).
siddon_reconstructor.mlem.__doc__ += REGULARIZATION_NOTE
siddon_reconstructor.osem.__doc__ += REGULARIZATION_NOTE
