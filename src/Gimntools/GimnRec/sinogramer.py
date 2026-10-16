import numpy as np
from numba import njit
import math

def generateRsectorAngle(detectors):
    """
        @brief Generates detector angles and rsector indices for a given number of detectors.

        @param detectors: int
            The number of detectors to generate angles and rsector indices for.

        @return: tuple (dict, dict)
            - detector: A dictionary mapping detector names (e.g., "detector-1") to their corresponding angles in degrees.
            - rsectors: A dictionary mapping detector names to their corresponding rsector indices (starting from 0).
    """
    angles= np.linspace(0,360,detectors,endpoint=False)
    detector={}
    rsectors={}
    dName = "detector-"
    for i,angle in enumerate(angles):
        name = dName+str(i+1)
        detector[name]=angle
        rsectors[name]=i
    return detector,rsectors


@njit
def convert_crystal_to_xy(crystal_id, n_columns=8, n_rows=8):
    """
    @brief Converts a crystal ID into (x, y) coordinates on a grid.

    This function calculates the (x, y) coordinates corresponding to a given crystal ID
    on an 8x8 grid. The crystal ID is used to determine the position on the grid,
    where x represents the column and y represents the row.

    @param crystal_id The ID of the crystal to be converted. It should be a non-negative integer.

    @return A NumPy array containing the (x, y) coordinates of the crystal.
            - out[0]: x-coordinate (column)
            - out[1]: y-coordinate (row)

    @note The function assumes that crystal IDs range from 0 to 63, corresponding to an
          8x8 grid (a total of 64 positions).
    """
    if (n_columns < 1 or n_rows < 1 or crystal_id < 0 or
            crystal_id >= n_columns * n_rows):
        raise ValueError("crystal_id outside segmented detector grid")
    out = np.zeros(2, dtype=np.int32)
    col = n_columns - 1 - (crystal_id % n_columns)
    row = crystal_id // n_columns
    out[0] = col
    out[1] = row
    return out


class Sinogramer:
    """
    @brief Class for generating and manipulating sinograms.

    This class manages the creation of sinograms based on the system configuration
    and provides methods for calculating slice indices and pixel positions.
    """

    sinogram = None
    system_conf = None

    def __init__(self, system_conf, matrix=np.asarray([24, 32, 32])):
        """
        @brief Initializes the Sinogramer class.

        This constructor takes as input the system configuration and the dimensions
        of the sinogram matrix.

        @param system_conf A dictionary containing the configuration of the system.
        @param matrix A NumPy array representing the dimensions of the sinogram.
                      Default is [128, 128, 64].
        """
        self.system_conf = system_conf
        self.matrix = matrix

        #print(system_conf)

        if self.system_conf['crystal_type'] == "monolithic":
            self.sinogram = np.zeros(matrix)
        elif self.system_conf['crystal_type'] == "segmented":
            n_angles    = int(((2 * system_conf['sipm_pixel_n'][0]) - 1) *self.system_conf['rotation_angles'])
            
            n_distances = int((2*system_conf['sipm_pixel_n'][0])-1)#*system_conf['detNum']/2 #int((2*system_conf['sipm_pixel_n'][0])-1*)
            n_slices    = int((2 * system_conf['sipm_pixel_n'][1]) - 1)

            self.matrix = np.asarray([int(n_slices),int(n_distances),int(n_angles)])
            #print(self.matrix)
            self.sinogram = np.zeros(self.matrix)

    def get_sinogram(self):
        """
        @brief Returns the current sinogram.

        @details
            This method provides access to the sinogram data stored in the object.
            The sinogram typically represents the raw projection data acquired in computed tomography (CT) or similar imaging modalities.

        @return
            The sinogram data associated with the object.
        """
        return self.sinogram
    

    
    def set_matrix(self,matrix):
        """
        @brief Sets the matrix attribute and initializes the sinogram array.

        @param matrix The shape or dimensions to be assigned to the matrix attribute.
                      This is typically a tuple specifying the desired shape.

        This method assigns the provided matrix shape to the object's matrix attribute
        and initializes the sinogram attribute as a NumPy array of zeros with the same shape.
        """
        self.matrix=matrix
        self.sinogram = np.zeros(self.matrix)
        
        
    def get_slice_mono(self, pair_pos_axis):
        """Slice index of a coincidence, for a monolithic (continuous) crystal.

        The two interaction coordinates along the slice axis are added, which
        is the standard direct/cross-plane rule: the mid-plane of the LOR is
        ``(a1 + a2)/2`` and the sum ``a1 + a2`` therefore samples that
        mid-plane on a grid of half the crystal pitch. The sum ranges over
        ``[-2*half, +2*half]`` with ``half = crystal_size[1]/2``, i.e. an
        axial field of view of ``crystal_size[1]`` in mid-plane coordinates.

        Parameters
        ----------
        pair_pos_axis : (a1, a2)
            Coordinates of the two interactions along the slice axis (mm).

        Returns
        -------
        int
            Slice index in ``[0, n_slices-1]``, or ``-1`` when the pair falls
            outside the axial field of view.

        .. note:: **Fixed in 0.4.0.** The index was previously computed with
           ``int()``, which truncates towards zero. A pair at the top of the
           axial range returned ``n_slices`` (``IndexError`` in
           :meth:`fill_sino`), and a pair just below the bottom returned 0
           instead of being rejected, folding out-of-field events into the
           first slice. The index is now floored and range-checked.
        """
        max_value = float(self.system_conf['crystal_size'][1]) / 2.0
        axial_span = 4.0 * max_value          # span of (a1 + a2 + 2*max_value)
        n_slices = int(self.sinogram.shape[0])
        bin_width = axial_span / n_slices
        prof = pair_pos_axis[0] + pair_pos_axis[1] + 2.0 * max_value
        idx = int(np.floor(prof / bin_width))
        if idx < 0 or idx >= n_slices:
            return -1
        return idx

    def get_slice_seg(self, pair_pos_axis):
        """
        @brief Calculates the slice index for segmented crystal type.

        This function computes a slice index based on the sum of the provided z-values.

        @param pair_z A tuple containing two z-values (float).

        @return The calculated slice index (int).
        """
        return pair_pos_axis[0] + pair_pos_axis[1]

    def get_pixel_position(self, distance, angle):
        """Map ``(r, phi)`` to the ``(radial_bin, angular_bin)`` sinogram cell.

        Conventions, shared with :mod:`Gimntools.GimnRec.normalization`,
        :class:`~Gimntools.GimnRec.reconstructor.siddon_reconstructor` and the
        pipelines:

        * the radial axis holds ``n_dist`` uniform bins covering
          ``[-R/2, +R/2]`` with ``R = system_conf['sino_range_mm']`` (falling
          back to ``crystal_size[1]``), so bin ``i`` is centred on
          ``(i + 0.5)*R/n_dist - R/2``;
        * the angular axis holds ``n_ang`` uniform bins covering ``[0, 180)``
          degrees, the parallel-beam range returned by
          :meth:`global_to_sinogram`, INDEPENDENTLY of the gantry arc.

        Returns
        -------
        (x, y) or None
            ``None`` when the LOR falls outside the sinogram.

        .. note:: **Fixed in 0.4.0.** Both indices were computed with
           ``int()``, which truncates towards zero rather than flooring. LORs
           with ``r`` in the one-bin-wide interval just below ``-R/2`` were
           therefore accepted into radial bin 0 instead of being rejected,
           producing an asymmetric count pile-up at the negative edge of the
           radial axis (the positive edge was correctly rejected). Both
           indices are now floored.
        """
        n_pixels_x = self.sinogram.shape[1]
        n_pixels_y = self.sinogram.shape[2]

        ref_dist = float(self.system_conf.get(
            'sino_range_mm', self.system_conf['crystal_size'][1])) / 2.0

        if not (np.isfinite(distance) and np.isfinite(angle)):
            return None

        y = int(np.floor((angle / 180.0) * n_pixels_y))
        normalized_distance = (distance + ref_dist) / (2.0 * ref_dist)
        x = int(np.floor(normalized_distance * n_pixels_x))

        if 0 <= x < n_pixels_x and 0 <= y < n_pixels_y:
            return (x, y)
        return None

    def fill_sino(self, distance, angle, slice_z):
        """Increment the sinogram cell addressed by ``(distance, angle, slice_z)``.

        Events falling outside the radial, angular or axial range are
        discarded (``slice_z < 0`` is the rejection code returned by
        :meth:`get_slice_mono`).
        """
        z = int(slice_z)
        if z < 0 or z >= self.sinogram.shape[0]:
            return
        pos = self.get_pixel_position(distance, angle)
        if pos is not None:
            self.sinogram[z, pos[0], pos[1]] += 1

    # Coordinate indices of the projection plane for each rotation axis.
    # The projection plane is the plane PERPENDICULAR to the rotation axis.
    _PLANE_AXES = {"z": (0, 1), "y": (0, 2), "x": (1, 2)}

    def global_to_sinogram(self, p1, p2, rotate_along):
        """Convert a LOR given by two global points into ``(r, phi)``.

        The LOR is projected onto the plane perpendicular to ``rotate_along``
        and described in the standard parallel-beam parametrization

            ``r = x*cos(phi) + y*sin(phi)``

        where ``phi`` is the angle of the line's NORMAL, wrapped into
        ``[0, 180)`` degrees. Because a line is undirected, ``(r, phi)`` and
        ``(-r, phi + 180)`` denote the same LOR; the wrap is therefore applied
        to ``phi`` first and ``r`` evaluated afterwards, which is exact.

        Parameters
        ----------
        p1, p2 : array-like of 3 floats
            The two interaction points in the global frame.
        rotate_along : {'x', 'y', 'z'}
            Rotation axis; the projection uses the other two coordinates.

        Returns
        -------
        (r, phi) : (float, float)
            Signed radial position in mm and projection angle in degrees,
            ``phi`` in ``[0, 180)``. Returns ``(nan, nan)`` if the two points
            coincide, in which case the LOR is undefined.

        .. note:: **Fixed in 0.4.0.** ``phi`` was previously computed as
           ``pi/2 + arctan(dy/dx)``. For a LOR exactly perpendicular to the
           first plane axis (``dx == 0``) this divided by zero, emitted a
           numpy warning and returned ``phi = 180`` degrees, which
           :meth:`get_pixel_position` then rejected -- the whole family of
           LORs at one particular orientation was silently dropped. Two
           coincident points returned ``nan`` and propagated it into the
           binning. ``arctan2`` on the normal vector removes both cases.
        """
        axis = str(rotate_along).lower()
        try:
            i0, i1 = self._PLANE_AXES[axis]
        except KeyError:
            raise ValueError(
                f"rotate_along must be 'x', 'y' or 'z', got {rotate_along!r}"
            ) from None

        u1, v1 = float(p1[i0]), float(p1[i1])
        u2, v2 = float(p2[i0]), float(p2[i1])

        du, dv = u2 - u1, v2 - v1
        if du == 0.0 and dv == 0.0:
            return float('nan'), float('nan')

        # Normal of the LOR: direction (du, dv) rotated by +90 deg.
        phi = np.arctan2(du, -dv) % np.pi      # wrapped into [0, pi)
        um, vm = 0.5 * (u1 + u2), 0.5 * (v1 + v2)
        r = um * np.cos(phi) + vm * np.sin(phi)
        return float(r), float(np.rad2deg(phi))
