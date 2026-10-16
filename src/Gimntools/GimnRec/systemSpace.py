import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R
from Gimntools.GimnRec.sinogramer import Sinogramer

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
        if (not isinstance(self.cols, (int, np.integer)) or
                not isinstance(self.rows, (int, np.integer)) or
                self.cols <= 0 or self.rows <= 0 or
                not np.isfinite([self.PichX, self.PichY]).all() or
                self.PichX <= 0 or self.PichY <= 0):
            raise ValueError("SiPM dimensions must be positive integers and pitches positive finite")
        # Column order and inverted row direction retain the legacy 8x8 mapping.
        self.positionsX = [float((i - (self.cols - 1) / 2) * self.PichX)
                           for i in range(self.cols)]
        self.positionsY = [float(((self.rows - 1) / 2 - j) * self.PichY)
                           for j in range(self.rows)]

        
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
        self.rotate_along = str(rotate_along).lower()
        self.detector_parameters = detector_parameters
        self.first_rotation = str(first_rotation).lower()

        # GimnToolsV2 FIX: `detectors` and `gantry_rotation` were CLASS
        # attributes (mutable, shared by ALL instances). Instantiating
        # systemSpace twice (e.g. PETSYS and then GATE) left stale entries
        # from the first in the second. They are now instance attributes.
        self.detectors = {}
        self.gantry_rotation = 0.0
        self.anguru = 0.0

        # This is where the rotations (and their definitions) happen.
        for detector_name, angle in detector_angles.items():
            self.detectors[detector_name] = R.from_euler(first_rotation, angle, degrees=True)
            #print(f"Creating detector: {detector_name} at the angle of {angle}")
        self.global_rot = R.from_euler(self.rotate_along, self.gantry_rotation, degrees=True)
        


        Sinogramer.__init__(self,detector_parameters)
        SiPM.__init__(self,
                      int(self.detector_parameters["sipm_pixel_n"][0]),
                      int(self.detector_parameters["sipm_pixel_n"][1]),
                      float(self.detector_parameters["sipm_pitch"][0]),
                      float(self.detector_parameters["sipm_pitch"][1]))
   


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
        # GimnToolsV2 FIX: use self.rotate_along instead of a hard-coded 'y'.
        # Before: R.from_euler('y', ...) ignored the configured axis, which
        # made the gantry rotation inconsistent with global_to_sinogram
        # whenever rotate_along != 'y' (e.g. GATE with rotate_along='z').
        self.global_rot = R.from_euler(self.rotate_along, self.gantry_rotation,
                                       degrees=True)


    def rotateBy(self, rotation: float):
        # GimnToolsV2 FIX: same as rotateGantry — honours self.rotate_along.
        self.global_rot = R.from_euler(self.rotate_along, rotation, degrees=True)


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
        # GimnToolsV2 FIX: global_to_sinogram(p1, p2, rotate_along) requires
        # the rotation axis. The old call (g1, g2) raised TypeError. Use the
        # axis configured on this instance.
        distance, angle = self.global_to_sinogram(g1, g2, self.rotate_along)

        return distance, angle
    
