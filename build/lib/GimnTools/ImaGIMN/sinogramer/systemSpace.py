import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R
from GimnTools.ImaGIMN.sinogramer import Sinogramer
#from sinogramer import Sinogramer

#from sinogramer import Sinogramer

class SiPM:
    def __init__(self, cols, rows, PichX, PichY):
        """
        Initializes a SiPM instance with specified dimensions and pitch values.

        @param cols: Number of columns in the SiPM.
        @type cols: int

        @param rows: Number of rows in the SiPM.
        @type rows: int

        @param PichX: Pitch value for the X positions.
        @type PichX: float

        @param PichY: Pitch value for the Y positions.
        @type PichY: float
        """
        self.cols = cols
        self.rows = rows
        self.PichX = PichX
        self.PichY = PichY
        self.positionsX = [0.0] * cols  # List to store X positions.
        self.positionsY = [0.0] * rows  # List to store Y positions.
        self.initialize()
    

    def initialize(self):
        """
        Initializes the positions of the SiPM based on the specified columns and rows.

        This method calculates the X and Y positions based on the pitch values and
        updates the positionsX and positionsY lists accordingly.
        """
        for i in range(self.cols):
            x = (i + 4) - 8
            if x < 0:
                x += 1

            if i < 4:
                x = x * self.PichX - self.PichX / 2.0
            else:
                x = x * self.PichX + self.PichX / 2.0
            
            self.positionsX[i] = x

        for j in range(self.rows):
            y = (j + 4) - 8
            if y < 0:
                y += 1

            if j < 4:
                y = -(y * self.PichY - self.PichY / 2.0)
            else:
                y = -(y * self.PichY + self.PichY / 2.0)

            self.positionsY[j] = y

        
    def get_positionsX(self):
        return self.positionsX

    def get_positionsY(self):
        return self.positionsY
@njit
def DetectorReference(event, crystal_size, pla_thickness, radius):
    """
    @brief Calculates the event position in the detector reference frame.

    This function takes as input four parameters:
    - event (3D np.array([x,y,z]) coming directly from PETSYS or GATE)
    - crystal_size (3D np.array([x_dim,y_dim,z_dim]) with the dimensions of the crystal)
    - pla_thickness (float) with the thickness of the PLA used directly in front of the crystal
    - radius (float) with the radius of the bore.

    @param event The 3D coordinates of the event.
    @param crystal_size The dimensions of the crystal.
    @param pla_thickness The thickness of the PLA in front of the crystal.
    @param radius The radius of the bore.

    @return The event position in the detector reference frame as a 3D np.array.
    """
    detector_referential = np.flip(event)
    offset = np.asarray([-crystal_size[0] / 2, 0, 0])
    pre_glob = detector_referential + offset
    detector_center = np.asarray([radius + pla_thickness + crystal_size[0] / 2, 0, 0])
    return pre_glob + detector_center


class systemSpace(Sinogramer,SiPM):
    """
    @brief Class representing the spatial configuration of a detector system.

    This class manages the rotation and positioning of detectors in a 3D space,
    allowing for calculations of event positions in both local and global reference frames.
    """
    anguru = 0
    # Dictionary that will store 'detector-name': rotationFunction
    detectors = {}
    # Dictionary that will store the detector name and the position in angles of each detector on the bore
    detector_parameters = {}

    # The gantry starts with no rotation, but it can be changed using rotateGantry() function
    gantry_rotation = 0.0
    # Defines the axis along which the rotation will occur
    rotate_along = None

    # Defines the rotation point of our system
    rotation_center = np.asarray([0, 0, 0])

    def __init__(self, detector_angles: dict, rotate_along: str, detector_parameters: dict, first_rotation :str = "z"):
        """
        @brief Initializes the systemSpace class.

        This constructor takes as input three parameters:
        - detector_angles (dict): Dictionary that stores 'detector-name': rotationFunction
        - rotate_along (str): Defines the axis along which the rotation will occur
        - detector_parameters (dict): Dictionary that stores the detector name and the position in angles of each detector on the bore

        @param detector_angles The angles for each detector.
        @param rotate_along The axis of rotation.
        @param detector_parameters Parameters related to the detectors.
        """
        self.rotate_along = rotate_along
        self.detector_parameters = detector_parameters
        self.first_rotation = first_rotation

        #aqui é onde ocorrem as rotações ou as definiçoes delas
        for detector_name, angle in detector_angles.items():
            self.detectors[detector_name] = R.from_euler(first_rotation, angle, degrees=True)
            #print(f"Creating detector: {detector_name} at the angle of {angle}")
        self.global_rot = R.from_euler(self.rotate_along, self.gantry_rotation, degrees=True)
        

        print(detector_parameters)

        Sinogramer.__init__(self,detector_parameters)
        SiPM.__init__(self,self.detector_parameters["sipm_pixel_n"][0],self.detector_parameters["sipm_pixel_n"][0],
                      self.detector_parameters["sipm_pitch"][0],self.detector_parameters["sipm_pitch"][0])
   


    def getGlobalPreRotation (self, event, detector_name):
        """
        @brief The function works as the getGlobalPosition, but instead of returning the data in global position, it returns
        the x,y,z position before the gantry rotation
      @return np.array([px, py, pz]) Event position in the global reference frame, before the rotation
        """
        return self.detectors[detector_name].apply( DetectorReference(event, self.detector_parameters["crystal_size"],
                              self.detector_parameters["pla_thickness"],
                              self.detector_parameters["radius"]))


    def getGlobalRotation(self,event):
        """
        @brief performs the gantry rotation for a given event, in the gantry space

        @return np.array([px,py,pz]) Event position in the global reference frame
        """
        return self.global_rot.apply(event)
    



    def getGlobalPosition(self, event, detector_name):
        """
        @brief Calculates the global position of an event.

        This function takes as input an event and the detector name where this event took place.
        It returns the event position in the global reference frame.

        @param event (np.array([px, py, pz])) Point of interaction of a given event inside the detector.
        @param detector_name (str) Name of the detector where the event occurred.

        @return np.array([px, py, pz]) Event position in the global reference frame.
        """
        
        return self.global_rot.apply(self.detectors[detector_name].apply(
            DetectorReference(event, self.detector_parameters["crystal_size"],
                              self.detector_parameters["pla_thickness"],
                              self.detector_parameters["radius"])))


    def getRotationMatrix(self):
        return self.global_rot

    def angleStep(self,angle):
        self.anguru += angle
        return self.anguru
    
    def angle(self):
        return self.anguru

    def rotateGantry(self, rotation: float):
        """
        @brief Rotates the gantry by a specified angle.

        This function updates the gantry rotation by adding the specified rotation angle.

        @param rotation The angle to rotate the gantry by.
        """
        self.gantry_rotation += rotation
        print("GANTRY ROTATION",self.gantry_rotation)
        print(self.rotate_along)
        self.global_rot = R.from_euler('y', self.gantry_rotation, degrees=True)
        # self.gantry_rotation += rotation
        # # Normaliza o ângulo para o intervalo [-180, 180)
        # self.gantry_rotation = self.gantry_rotation % 360.0
        # if self.gantry_rotation < 0:
        #     self.gantry_rotation += 360.0
        # self.global_rot = R.from_euler('y', self.gantry_rotation, degrees=True)


    def rotateBy(self, rotation: float):
        self.global_rot = R.from_euler('y', rotation, degrees=True)


    def getAngleOfRotation(self):
        """
        @brief Retrieves the current angle of rotation of the gantry.

        @return The current gantry rotation angle.
        """
        return self.gantry_rotation

    def CalculateSinogram(self, event1, detector1_name, event2, detector2_name):
        """
        @brief Calculates the sinogram for two events.

        This function retrieves the global positions of two events and computes the
        distance and angle between them.

        @param event1 The first event.
        @param detector1_name The detector name for the first event.
        @param event2 The second event.
        @param detector2_name The detector name for the second event.
        """
        
        
        g1 = self.getGlobalPosition(event1, detector1_name)
        g2 = self.getGlobalPosition(event2, detector2_name)

        #print(g1)
        #print(g2)
        distance, angle = self.global_to_sinogram(g1, g2)
        
        return distance, angle
    
