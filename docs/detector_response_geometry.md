# Detector response geometry

The legacy 2D Siddon operator traces parallel rays in a plane perpendicular
to the rotation axis. Its finite radial-bin option samples parallel offsets.
The existing radial PSF applies a sinogram convolution after forward projection
and its transpose before backprojection. MLEM and OSEM use the same transpose
to compute sensitivity from normalization, separately for each OSEM subset.
FBP uses its own filtering path.

## Geometry and available data

* The example PETSYS configuration describes two opposing blocks with an
  8 × 8 grid on 3.36 mm pitch. The `crystal_size` vector
  `[20, 26.88, 26.88]` is the block envelope. The task geometry specifies
  3 × 3 mm segment faces and 20 mm depth.
* `DetectorReference` flips the input coordinates into detector coordinates.
  At local depth zero the outward radial coordinate is
  `radius + pla_thickness`. The example fields give 37 mm to the crystal
  entrance, or 74 mm between opposing entrances. This differs from both a
  35 mm entrance radius and the quoted 75 mm entrance-to-entrance distance.
  A quantitative model needs an independently confirmed crystal entrance
  radius and a definition of the reference surface used by the calibration.
* For the example rotation axis `y`, `global_to_sinogram` projects onto
  `(x, z)`. PETSYS applies the configured detector-local z rotation, maps
  each chip ID to a detector, then applies the negative acquisition angle
  to express the LOR in the object's frame. The example uses 12 positions
  and the loader applies a 450–650 keV energy window to both hits.
* The GATE CrystalID path reads crystal, sector and run IDs before
  histogramming. The crystal column reverses the modulo of the ID and the
  row is the integer quotient. The global-position paths use recorded hit
  coordinates. The PETSYS reader uses chip IDs for detector identity and
  SiPM positions for hit locations. Assigning a PETSYS event to a physical
  segment requires a verified mapping of those positions to the crystal grid.
* Both formats return a histogram and a table with only distance, angle
  and slice. Binning into signed radial position, normal angle modulo 180°
  and axial slice can combine multiple detector pairs or acquisition positions.
  The resulting histogram cannot recover those event identities.

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
