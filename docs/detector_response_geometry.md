# Detector response geometry

The legacy 2D Siddon operator traces parallel rays in a plane perpendicular
to the rotation axis. Its finite radial-bin option samples parallel offsets.
The existing radial PSF applies a sinogram convolution after forward projection
and its transpose before backprojection. MLEM and OSEM use the same transpose
to compute sensitivity from normalization, separately for each OSEM subset.
FBP uses its own filtering path.

## Geometry and available data

* PETSYS **Build sinogram** reads the detector configuration and the
  **Angles / rsectors** table from the UI at build time. It passes the
  geometry, detector orientations and chip-to-detector mapping to
  `coincidence_to_lor`. This is the geometry source for a response model
  that follows the acquisition setup.
* `DetectorReference` flips the input coordinates into detector coordinates.
  The local depth-zero plane used to build the sinogram is at
  `radius + pla_thickness` from the center. The values 35 mm and 2 mm
  in the example JSON give 37 mm in that example. With opposing detectors,
  this gives a 74 mm separation between their reference planes. The UI
  settings for the acquisition determine the actual value used by the code.
* The example PETSYS configuration describes an 8 × 8 grid on 3.36 mm
  pitch. The `crystal_size` vector `[20, 26.88, 26.88]` is the block
  envelope. A finite-crystal model also needs the active segment face
  dimensions (3 × 3 mm for the specified scanner), which are distinct from
  the grid pitch.
* For the example rotation axis `y`, `global_to_sinogram` projects onto
  `(x, z)`. PETSYS applies the configured detector-local z rotation, maps
  each chip ID to a detector, then applies the negative acquisition angle
  to express the LOR in the object's frame. The example uses 12 positions
  and the loader applies a 450–650 keV energy window to both hits.
* The GATE CrystalID path reads crystal, sector and run IDs before
  histogramming. The crystal column reverses the modulo of the ID and the
  row is the integer quotient. The global-position paths use recorded hit
  coordinates. The PETSYS reader uses chip IDs for detector identity and
  ROOT SiPM positions directly for hit locations. The configured pitch
  and grid dimensions define the nominal SiPM lattice. For a pair-aware
  model, the reader can compare each hit with that lattice in the matching
  local frame and retain its segment index before binning. That comparison
  needs to establish whether the input positions represent discrete segment
  centers or continuous hit estimates.
* Both formats return a histogram and a table with only distance, angle
  and slice. Binning into signed radial position, normal angle modulo 180°
  and axial slice can combine multiple detector pairs or acquisition positions.
  The resulting histogram cannot recover those event identities. The UI
  keeps the detector configuration and sinogram angle axis with the
  acquisition. The original detector-angle table, chip mapping and pair
  identities are available during the build and would need to be carried
  forward for pair-aware reconstruction.

## DOI model

The standalone quadrature treats first interaction as exponential in path
length at 511 keV. A finite depth interval is conditioned on interaction,
and its probability is returned separately for acceptance weighting. An
explicit segment face width and entrance coordinate bound the interval at
the side wall when an oblique ray exits early. The calculation uses millimetres
and requires an explicit attenuation coefficient in inverse millimetres.

The resulting weights describe first interaction only. Compton redistribution,
wrong-crystal assignment, non-collinearity and positron range are outside the
model. Synthetic tests verify the quadrature numerically. Independent GATE
and experimental validation require separate datasets.

The operator uses the geometry and discrete object-frame poses captured
when the ROOT sinogram is built. It enumerates all transverse segment pairs
on two opposing heads, including pairs with zero measured counts. Nominal
face-center LORs determine the original sinogram bins. When a nominal LOR
lands numerically on an internal bin edge, its response is shared across
the tied bins. At the angular seam, radial parity follows the signed LOR
convention. Each pair contributes weighted Siddon rays across both
interaction volumes. The calculation uses transverse face quadrature and
depth quadrature at both ends, corrects the first-interaction exponential
for each sampled ray's incidence, and accepts rays whose entrance points
lie within both segment faces. The 2D line measure weights the product
of the first-interaction densities by the inverse endpoint separation.
An encoder offset and COR corrections move the rays in the reconstruction
frame while preserving the measured bin labels.

This is an effective sinogram model for slice-wise 2D data. Its assumptions
are equal exposure per configured pose, depth zero at `radius + pla_thickness`
in the acquisition geometry, and projection of segment pairs into the slice
plane. It omits axial obliquity, inter-slice transport, time of flight and
per-segment efficiency differences. In the example JSON, a 35 mm bore
reference and 2 mm PLA place the reference planes 74 mm apart. The 75 mm
value quoted for the scanner should be reconciled with the physical
reference surface when setting up the acquisition. A measured flood
efficiency can enter the public EM API as diagonal normalization.
The legacy geometric-acceptance option would count detector acceptance
twice with DOI.

PETSYS requires a segmented lattice aligned to the slice axes. GimnROOT
maps `channelID` to row `I` and column `J` with
`SiPM::CorrectChannelToPixelPs`. Before alignment, the stored coordinate
conventions are `X = (J−3.5) pitch_x` and `Y = (3.5−I) pitch_y` for the
8 × 8 grid. `ProcessAcquisition.C` optionally subtracts the chip's
`alignmentCorrection[x,y]` from those coordinates. The Anger-coordinate
branches are separate and do not enter this segmented model.

GimnROOT commit `31b862f` assigns `tse1.posJ = first_j2` after reading
`first_j1` and `first_j2` independently. In affected files, `posJ_1`
contains the column of hit 2. GimnTools derives both IDs from
`channelID_1/2`, cross-checks `posI_1/2` and `posJ_2` when present, and
does not use `posJ_1`. The upstream repository is unchanged.

GimnTools compares the decoded crystal centers with the stored measurements
to recover the constant per-chip translation. It checks the original
1%-of-pitch residual tolerance after the translation and verifies each
observed transverse center is stable within float32 storage precision.
It keeps the observed encoded centers for measured bin labels, including
the configured local Z rotation and coordinate reversal. This follows
GimnROOT's local conversion from `(X,Y,depth)` to `(depth,Y,X)` after its
chip rotation. GimnTools uses the raw `SiPMPosX/Y` branches and applies
its own configured detector placement and object angle once. It does not
reuse the already-transformed `globalPos` or `rSino` branches. Physical
quadrature uses the configured nominal crystal volumes. If the scanner
survey requires a physical displacement, specify it through the scanner
geometry rather than reusing an image-alignment offset. Data without the
GimnROOT channel branches retain the nominal-grid check. Inconsistent
channel/index data, continuously changing translations and same-head
coincidences remain incompatible with DOI.

Build a raw, unrebinned sinogram. The desktop derives the PETSYS pose
count from the recorded `angleGantry_1` values. For GATE, use `CrystalID`
mode. The detector model uses the observed run IDs with their original
angular step and assumes equal exposure across those runs. A missing run
ID cannot distinguish an absent exposure from an exposure with zero
recorded events. MLEM/OSEM report an error for positive counts in bins
with neither modeled sensitivity nor background.

[GimnROOT coincidence writer](https://github.com/michelraed/GimnRoot/blob/31b862ff72ca9aefc15a54c32990568c29576871/source/92-TomoProcessing/ProcessAcquisition.C#L2748)
· [GimnROOT channel map](https://github.com/michelraed/GimnRoot/blob/31b862ff72ca9aefc15a54c32990568c29576871/source/2-histograms/SiPM.C#L332)
· [GimnROOT coordinate transforms](https://github.com/michelraed/GimnRoot/blob/31b862ff72ca9aefc15a54c32990568c29576871/source/91-sinogram/GantryPositions.C#L170)

The [synthetic phantom example](../examples/detector_response_phantom.py)
compares the four operator configurations. Its reconstructed-source widths
describe the reconstruction at the chosen grid, iterations and count level.
Scanner validation needs real-data calibration and independent GATE data.

## Spatially variant PSF

Supply an `.npz` with the following arrays. All axes are finite, strictly
increasing 1D arrays in millimetres. The origin and axis orientation match
the reconstructed slice coordinates used by Siddon. For rotation about
scanner `y`, the slice axes are scanner `x` and `z`.

| Field | Meaning |
| --- | --- |
| `x_mm`, `y_mm` | Calibrated source locations along the two image axes |
| `dx_mm`, `dy_mm` | Offsets from each source to the support of its response |
| `kernels` | Nonnegative weights with shape `(len(y_mm), len(x_mm), len(dy_mm), len(dx_mm))` and unit sum at every source |
| `provenance` | Scalar Unicode description of the GATE run, reference model, geometry, calibration and kernel derivation |
| `doi_included` | Scalar Boolean stating whether the LUT includes DOI |

The LUT represents probability mass for an **image-domain response within
the system matrix**. Interpolate between source locations bilinearly and
use the nearest calibrated edge for sources beyond the calibration grid.
Deposit sub-voxel offsets bilinearly into the target grid. Discard mass
that leaves the image domain. Kernels may be asymmetric. If ordered as
`A_D C`, with DOI projection `A_D` and LUT response `C`, the
backprojection is `Cᵀ A_Dᵀ`. MLEM and each OSEM subset use the corresponding
transpose to compute sensitivity. This response is applied before each
projection, and its transpose is part of backprojection.

For composition with DOI, derive the LUT as a **residual** relative to
the modeled DOI physics. A LUT calibrated with DOI already included is
accepted only as the standalone spatially variant PSF. Qualify each LUT
against a calibration independent of the data used to derive it. The
toolkit marks imported LUT reconstructions as exploratory.

## Use

- In the GUI, build the raw sinogram first, select MLEM or OSEM, then pick
  **DOI-aware**, **Spatially variant PSF**, or **DOI + residual SV response**
  in **Detector Response Modeling**. Select **LYSO**, **BGO**, or **Custom**.
  The starting material, attenuation override and segment width come from
  the selected sinogram's detector configuration. Existing profiles without
  these fields start with LYSO and a 3 mm segment face. A manually entered
  coefficient overrides the preset. Each reconstruction keeps its response
  settings in the output metadata.
- Either CLI enables DOI with `--crystal-material LYSO` or
  `--crystal-material BGO`. Override the material coefficient and active
  width with `--doi-mu-mm-inv` and `--doi-face-mm`.
  `--doi-face-order` and `--doi-depth-order` default to 3 on the CLI.
  The desktop defaults to **Auto** for the face order. Auto chooses an order
  from 3 to 12 based on the face intervals and voxel width. It is also
  available via `--doi-face-order auto`. Use
  `--sv-psf-lut path.npz` for SV-PSF alone or alongside DOI. Legacy EM
  and the existing radial PSF are separate options.
- Python callers can pass `ResponseSettings` as `detector_response` to
  `process_file_v2` or `process_file`. To call MLEM/OSEM directly, pass
  `geometry_out={}` to `coincidence_to_lor`, then build
  `DetectorResponse(reconstructor, angles_deg, settings, geometry_out)`
  and pass it as `response` to the EM method. You can use the same
  operator for MLEM and OSEM on that reconstructor and angular grid.
- CLI NPZ outputs retain settings, acquisition geometry, input paths and
  LUT SHA-256 and provenance via `geometry_json.detector_response`. Image
  exports include the response for MLEM/OSEM in `GimnToolsDetectorResponse`.

The numerical quadrature and sparse methods impose size limits and raise
an error when those limits are exceeded. Use a smaller reconstruction
grid or smaller quadrature orders and check convergence before comparing
quantitative results.


## Crystal presets and saved profiles

| Preset | Representative composition | Density (g/cm³) | μ at 511 keV (mm⁻¹) |
| --- | --- | ---: | ---: |
| LYSO | Lu₁.₈Y₀.₂SiO₅ | 7.10 | 0.08184 |
| BGO | Bi₄Ge₃O₁₂ | 7.13 | 0.09629 |

Coefficients were calculated by mass-weighting elemental total attenuation
from xraylib 4.3.0 at 511 keV and multiplying by the nominal density.
Use a coefficient for your crystal's measured composition and density if
they differ. Material selection changes the analytical and DOI response.
The dimensions of the detector still come from the geometry.

The **PETSYS profile** and **GATE profile** pickers show only profiles
of the matching acquisition format from the shared SQLite database. On
startup the application restores the last loaded profile for each format.
For older databases it uses the exact format name, or a sole profile for
that format. Loading a PETSYS profile restores its calibration if saved.
A newly built sinogram keeps the detector configuration that produced it,
even if you later switch the active profile.

[xraylib documentation](https://github.com/tschoonj/xraylib/wiki)

## Full analytical response and automatic LUT generation

The full operator models collinear photons and first interaction at
511 keV in the transaxial plane. For a ray that crosses a crystal after
traversing a total scintillator length `L_pre`, its interaction probability
in a further chord of length `L_c` is
`exp(-μ L_pre) (1 - exp(-μ L_c))`. Gap lengths have no attenuation in this
model. The implementation integrates its differential form
`μ exp(-μ (L_pre + ℓ)) dℓ` at both interaction points. Slab intersections
include lateral entry or exit and preceding neighboring crystals.
Interaction locations are integrated with Gauss–Legendre quadrature in the
transverse and depth coordinates of both segments. The change of variables
to a line and its two interaction locations contributes the inverse
endpoint separation. The line measure is normalized by π and by the
voxel area. Siddon integrates the ray over each voxel. Columns retain
geometrical and interaction efficiency, including losses from the
measured bin range. The resulting CSR matrix supports its exact
transpose. Relative pose exposure is normalized to unit sum, with equal
exposure by default. Python callers can supply a `pose_exposure` list in
sorted `object_angles_deg` order in the acquisition metadata.

Select **Analytical detector response** in either UI, or use
`--analytical-response` in either reconstruction CLI. It uses the
configured material and face width unless you override them. Rays contribute
only through the gap between the detector faces for each pose. The requested
image matrix and FOV are retained, including pixels with limited or zero
sensitivity. Converge the face and depth orders on the
grid you intend to use. In a synthetic check with central and displaced
sources, the relative projection differences versus order 9 dropped
from 7.0–7.8% at order 3 to 1.5–1.6% at order 5 for both coordinates.

Use **Generate analytical LUT…** in Tk or the **Analytical LUT** tab in Qt
to sample this response automatically. The generator fits each sampled
projection column using a nonnegative unit-mass kernel composed with
Siddon. The reference projection is multiplied by the ratio of the
analytical and Siddon projections of a uniform image, bin by bin.
This lets the reference represent the detector's acceptance even in bins
with zero support. The fit minimizes a weighted relative projection
residual and a kernel gradient penalty, with a free
nonnegative amplitude eliminated analytically. `kernel_stride` sets the
kernel sampling in integer voxel steps. The saved NPZ includes the
fitted amplitude, analytical efficiency, relative error, processing time
and provenance. The NPZ also stores the matched reference acceptance
and its grid. To reuse this LUT with a different matrix, voxel size or
image field, build the sinogram from the matching acquisition and choose
the target image grid. GimnTools deposits the kernel offsets in mm onto the
new grid and recomputes the uniform-field bin acceptance. The sinogram
grid, detector setup, poses and COR must still match. The fit error in the
saved file measures the original grid. Refit on the target grid to measure
the error there. The normalized kernel alone does not carry efficiency.
The preview shows a sampled kernel, X/Y marginal profiles, centroid,
reflection asymmetry, fit error and FWHM where both half-height crossings
are sampled. Fits with large projection error should use the direct
analytical operator. FWHM here describes the fitted kernel on a chosen
grid.

Generated LUTs include DOI and load as a standalone SV-PSF. A nontrivial
residual LUT needs independent physics or a calibration relative to a DOI
reference. The generator only exports the full response. It assumes no
positron range, noncollinearity, Compton redistribution or energy-window
acceptance beyond first interaction. The numerical checks use synthetic
data. Comparison with independent GATE or scanner data remains necessary
for physical validation.

Gravel et al. model position-dependent, anisotropic, asymmetric PSFs within
PET reconstruction. Their study characterizes kernels on a limited-angle
dual-panel scanner with simulated data and uses TOF DIRECT. This 2D
implementation uses the same image-domain representation idea and fits
kernels to first-interaction projection probabilities for rotating
detectors. The paper's performance results apply to its own scanner and
protocol.

[Strul et al. (2003)](https://doi.org/10.1088/0031-9155/48/8/302)
· [Iriarte et al. (2009)](https://doi.org/10.1088/0031-9155/54/7/004)
· [Pratx and Levin (2011)](https://doi.org/10.1088/0031-9155/56/13/018)
· [Gravel et al. (2019)](https://doi.org/10.1088/1361-6560/ab4914)


### Interpreting dark patches in a reconstruction

The analytical projector uses discrete quadrature rays through the crystal
faces and depths. The **rays/bin** setting belongs to the conventional
Siddon projector. A finer image grid can show sampling variations more
clearly when the analytical quadrature order is kept fixed. The **Auto**
face order tracks voxel size. Depth order remains a separate setting.

The SV-PSF path spreads each pixel contribution through the LUT before
projecting. Smoother appearance can reflect that mixing. Low counts and
OSEM subset updates can also drive some voxels to low or zero values.
Compare the same slice and intensity limits, and compare OSEM to MLEM
with the same analytical model to isolate the effect of subsets. The
**EM relaxation** control can damp an OSEM update. The default of 1 uses
the full update.
