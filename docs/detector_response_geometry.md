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

PETSYS requires a segmented lattice aligned to the slice axes. The loader
checks that accepted transverse SiPM coordinates are within 1% of a pitch
of a nominal center and that the hits are on opposite heads. For GATE,
build the sinogram in `CrystalID` mode. DOI requires all configured poses
in the input ROOT and a raw, unrebinned sinogram. MLEM/OSEM report an error
for positive counts in bins with neither modeled sensitivity nor background. A recording
with continuous SiPM hit estimates needs a different acquisition model.

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
  in **Detector Response Modeling**. Supply the active segment width, a
  verified 511 keV linear attenuation coefficient in mm⁻¹, and the LUT file
  as applicable. Each successful reconstruction keeps its response metadata.
- Either CLI enables DOI with `--doi-mu-mm-inv` and `--doi-face-mm`.
  `--doi-face-order` and `--doi-depth-order` default to 3. Use
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
