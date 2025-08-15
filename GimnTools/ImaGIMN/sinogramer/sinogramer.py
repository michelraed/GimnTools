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
def convert_crystal_to_xy(crystal_id):
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
    number_of_rows = 8
    number_of_columns = 8
    out = np.zeros(2, dtype=np.int32)
    #out[1] = crystal_id // number_of_rows
    #out[0] = (number_of_columns - 1) - (crystal_id - out[1] * number_of_rows)
    

    col  = 7 - (crystal_id%8)
    row = crystal_id//8
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

        print(system_conf)

        if self.system_conf['crystal_type'] == "monolithic":
            self.sinogram = np.zeros(matrix)
        elif self.system_conf['crystal_type'] == "segmented":
            n_angles    = int(((2 * system_conf['sipm_pixel_n'][0]) - 1) *self.system_conf['rotation_angles'])
            
            n_distances = int((2*system_conf['sipm_pixel_n'][0])-1)#*system_conf['detNum']/2 #int((2*system_conf['sipm_pixel_n'][0])-1*)
            n_slices    = int((2 * system_conf['sipm_pixel_n'][1]) - 1)

            self.matrix = np.asarray([int(n_slices),int(n_distances),int(n_angles)])
            print(self.matrix)
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
        """
        @brief Calculates the slice index for mono crystal type.

        This function computes a slice index using the provided pair of z-values.

        @param pair_z A tuple containing two z-values (float).

        @return The calculated slice index (int).
        """
        # print ("----------------------------Get Slice Mono --------------------------")
        max_value = self.system_conf['crystal_size'][1] / 2
        new_axis = 4 * max_value
        multiplier = new_axis / self.sinogram.shape[0]
        # print(f"pair pos {pair_pos_axis} ")
        # print(f"max value {max_value}")
        # print(f"multiplier {multiplier} ")
        prof = pair_pos_axis[0] + max_value + pair_pos_axis[1] + max_value
        out = int(prof / multiplier)

        # print ("----------------------------Get Slice Mono --------------------------")

        return out

    def get_slice_seg(self, pair_pos_axis):
        """
        @brief Calculates the slice index for segmented crystal type.

        This function computes a slice index based on the sum of the provided z-values.

        @param pair_z A tuple containing two z-values (float).

        @return The calculated slice index (int).
        """
        return pair_pos_axis[0] + pair_pos_axis[1]

    def get_pixel_position(self, distance, angle):
        """
        @brief Calculates the pixel position in a 128x128 image based on distance and angle.

        This function maps the distance and angle to pixel coordinates (x, y) in an image.
        The angle varies from 0 to 360 degrees, and the distance varies from -13.44 to 13.44.

        @param distance The distance value (float) to be mapped.
        @param angle The angle value (float) in degrees to be mapped.

        @return A tuple (x, y) representing the pixel coordinates in the image.
                Returns (-1, -1) if the inputs are out of bounds.
        """
        # Define the image dimensions
        # print ("----------------------------------------------------")
        # print (f"Distance {distance}")
        # print (f"angle {angle}")

        n_pixels_x = self.sinogram.shape[1]
        n_pixels_y = self.sinogram.shape[2]
        
        ref_dist = self.system_conf['crystal_size'][1] / 2
        # Map angle to y-coordinate
        # Angle range: 0 to 360 maps to 0 to 127 (y-coord)
        y = int((angle / self.system_conf['arround']) * (n_pixels_y))#-1

        # Map distance to x-coordinate
        # Distance range: -13.44 to 13.44 maps to 0 to 127 (x-coord)
        # Normalize the distance to a range of 0 to 1
        normalized_distance = (distance + ref_dist) / (2 * ref_dist)
        x = int(normalized_distance * (n_pixels_x ))


        # print (f"on Pixel Position x {x} , y {y}, npixelsX {n_pixels_x}, npixelsy {n_pixels_y}")
        # # Check if coordinates are within bounds
        # print ("----------------------------------------------------")

        if 0 <= x < n_pixels_x and 0 <= y < n_pixels_y:
            return (x, y)
        else:
            return None  # Return an invalid position if out of bounds



    def fill_sino(self, distance, angle, slice_z):
        """
        @brief Fills the sinogram for mono crystal type based on distance, angle, and slice index.

        This function calculates the appropriate slice index and pixel position,
        then increments the sinogram at those coordinates.

        @param distance The distance value (float) to be used for filling.
        @param angle The angle value (float) to be used for filling.
        @param slice_z The slice index (int) to be used for filling.
        """
        z = slice_z
        pos = self.get_pixel_position(distance, angle)
        if pos is not None:
            x = pos[0]
            y = pos[1]
            #print(f"x {x} , y {y} , z {z}")
            self.sinogram[z,x,y] += 1
    


    def global_to_sinogram(self, p1, p2, rotate_along):
        """
        @brief Converts the global coordinates of two points to sinogram coordinates in a specified rotation plane.

        This function calculates the distance parameter (r) and the angle (phi) corresponding to the projection of points p1 and p2 onto a sinogram,
        considering rotation along one of the main axes ('x', 'y', or 'z'). The calculation is performed for the specified rotation plane,
        using the midpoint of the coordinates and the slope of the line connecting the points.

        @param p1: tuple or list
            The first point in global space, represented as a tuple or list of three coordinates (x, y, z).
        @param p2: tuple or list
            The second point in global space, represented as a tuple or list of three coordinates (x, y, z).
        @param rotate_along: str
            The axis along which the rotation is considered for the sinogram calculation. Can be 'x', 'y', or 'z'.

        @return: tuple (r, phi)
            r (float): Perpendicular distance from the center of the system to the midpoint of the line connecting p1 and p2, in the rotation plane.
            phi (float): Projection angle in degrees, relative to the specified rotation axis.

        @details
            - For each rotation axis, the two relevant coordinates for the plane perpendicular to the axis are used.
            - The angle phi is calculated in radians and converted to degrees.
            - The parameter r represents the position of the projection line in the sinogram.
            - The function assumes that p1 and p2 are in the format (x, y, z).
        """
        if rotate_along == "z":

            xm = (p2[0]+p1[0])/2
            ym= (p2[1]+p1[1])/2
            
            dx2 = p2[0]-p1[0]
            dy2= p2[1]-p1[1]

            theta = np.arctan(dy2/dx2)
            phi =  ((np.pi/2)+theta)
            r =xm*np.cos(phi)+ym*np.sin(phi)
            phi = np.rad2deg(phi)

        if rotate_along == "y":
         
            xm = (p2[0]+p1[0])/2
            ym= (p2[2]+p1[2])/2
            
            dx2 = p2[0]-p1[0]
            dy2= p2[2]-p1[2]

            theta = np.arctan(dy2/dx2)
            phi =  ((np.pi/2)+theta)
            r =xm*np.cos(phi)+ym*np.sin(phi)
            phi = np.rad2deg(phi)


        if rotate_along == "x":
            xm = (p2[1]+p1[1])/2
            ym= (p2[2]+p1[2])/2
            
            dx2 = p2[1]-p1[1]
            dy2= p2[2]-p1[2]

            theta = np.arctan(dy2/dx2)
            phi =  ((np.pi/2)+theta)
            r =xm*np.cos(phi)+ym*np.sin(phi)
            phi = np.rad2deg(phi)

        return r, phi

