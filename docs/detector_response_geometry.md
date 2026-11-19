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

## Phase A model

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

A matched extension can construct sparse Siddon rows from finite-face
quadrature for each calibrated detector pair and acquisition position, including
eligible pairs with zero observed counts for sensitivity. An effective
sinogram route would need an explicit mapping of those rows into acquisition
bins, with the transpose of that same mapping used in backprojection.
A residual GATE response needs independent provenance to establish whether
its kernels already include DOI.
