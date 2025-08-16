from GimnTools.ImaGIMN.sinogramer.sinogramer import convert_crystal_to_xy
from GimnTools.ImaGIMN.sinogramer.sinogramer import Sinogramer
from GimnTools.ImaGIMN.sinogramer import systemSpace
from GimnTools.ImaGIMN.sinogramer import generateRsectorAngle

from scipy.spatial.transform import Rotation as R

import uproot as up
import pandas as pd
import numpy as np




def coincidence_to_lor(file_name:str,detector_config:dict,detectors_angles:dict,parameters,matrix=[64,64,64],rsectors={ 0 : "detector-1",1 : "detector-2"}):
    """
        @brief
        Processes coincidence event data from PET (Positron Emission Tomography) experiments and converts them into Lines of Response (LORs) for sinogram generation.
        @details
        This function reads coincidence event data from a ROOT file, processes the events according to the specified detector configuration, and computes the corresponding LORs. The function supports different processing modes based on the detector configuration, such as using crystal IDs or global positions provided by simulation tools like GATE or PETSYS. For each event, it calculates the distance and angle of the LOR in the sinogram space, determines the corresponding slice, and fills the sinogram accordingly. The function also handles gantry rotation by updating the system geometry as needed for each run. The results are returned as a sinogram array and a DataFrame containing the LOR parameters for further analysis or visualization.
        @param file_name
            Path to the ROOT file containing the coincidence event data.
        @param detector_config
            Dictionary containing the configuration of the detector system, including processing mode, number of detectors, rotation angles, and other relevant parameters.
        @param detectors_angles
            Dictionary specifying the angular positions of the detectors.
        @param parameters
            List or structure specifying which parameters to extract from the event data.
        @param matrix
            List of three integers specifying the dimensions of the sinogram matrix (default: [64, 64, 64]).
        @param rsectors
            Dictionary mapping rsector IDs to detector names (default: {0: "detector-1", 1: "detector-2"}).

        @return
            Tuple containing:
                - sin: Numpy array representing the filled sinogram.
                - df: Pandas DataFrame with columns 'distance', 'angle', and 'slice', containing the LOR parameters for each processed event.
        @note
        The function supports multiple processing modes, including:
            - "CrystalID": Uses crystal IDs to compute event positions.
            - "GateGlobalPositions": Uses global positions provided by GATE.
            - "PaulGlobalPosition": Uses global positions with a specific axis convention.
        The function also visualizes the sinogram for certain processing steps and prints debug information for tracking the computation.

    """

    import matplotlib.pyplot as plt

    PETSYS = systemSpace(detectors_angles,"y",detector_config,"y")
    file = up.open(file_name)
    coincidences = file['Coincidences'].arrays(parameters,library='np')
    events = (coincidences['globalPosX1'].shape[0])
    
    if (detector_config["process_as"] =="GateGlobalPositions") or  (detector_config["process_as"] =="PaulGlobalPosition"):
        PETSYS.set_matrix(matrix)
    #track this guy here, depending on the experiment it can change
    angle_step = (360/detector_config['rotation_angles'])/detector_config['detNum']
    previous =0

    print(f"angle step {angle_step}")
    #sinogram DataFrame
    df = {'distance':[],
       'angle':[],
       'slice':[]}
    previous =0 
    cum = 0
    print(events)

    for event in range(events):
        if detector_config["data_from"] == "GATE":
            # Loads data to variables 
            crystalID1 = coincidences['crystalID1'][event] 
            crystalID2 = coincidences['crystalID2'][event]
            globalPosX1= coincidences["globalPosX1"][event]
            globalPosY1= coincidences["globalPosY1"][event]
            globalPosZ1= coincidences["globalPosZ1"][event]
            globalPosX2= coincidences["globalPosX2"][event]
            globalPosY2= coincidences["globalPosY2"][event]
            globalPosZ2= coincidences["globalPosZ2"][event]
            sinogramS =  coincidences["sinogramS"][event]
            sinogramTheta =  coincidences["sinogramTheta"][event]
            rSector1= coincidences["rsectorID1"][event]
            rSector2= coincidences["rsectorID2"][event]
            runID = coincidences["runID"][event]

            if detector_config["process_as"] =="CrystalID":
                if runID == previous:
                    # takes the crystalID

                    X1,Y1  = convert_crystal_to_xy(crystal_id=crystalID1)
                    X2,Y2 = convert_crystal_to_xy(crystal_id=crystalID2)
                    
                    # conversts crystalID into an event position
                    event1_positions = np.asarray([-PETSYS.positionsX[X1],-PETSYS.positionsY[Y1],0])
                    event2_positions = np.asarray([-PETSYS.positionsX[X2],-PETSYS.positionsY[Y2],0])


                    p1 = PETSYS.getGlobalPreRotation(event1_positions, rsectors[rSector1])
                    p2 = PETSYS.getGlobalPreRotation(event2_positions, rsectors[rSector2])
                    

                    g1 = PETSYS.getGlobalRotation(p1)
                    g2 = PETSYS.getGlobalRotation(p2)
                    

                    distance , angle= PETSYS.global_to_sinogram(g1, g2,"y")


                    slice_z=int(PETSYS.get_slice_seg((Y1,Y2)))


                    df['angle'].append(angle)
                    df['distance'].append(distance)
                    df["slice"].append(slice_z)

                    PETSYS.fill_sino(distance,angle,slice_z)
                else:

                    cum += angle_step
                    previous = runID
         
                    # ESTE CARA FAZ O GANTRY ROTACIONAR

                    PETSYS.rotateGantry(-angle_step)

                    X1,Y1 = convert_crystal_to_xy(crystal_id=crystalID1)
                    X2,Y2 = convert_crystal_to_xy(crystal_id=crystalID2)


                    event1_positions = np.asarray([-PETSYS.positionsX[X1],-PETSYS.positionsY[Y1],0])
                    event2_positions = np.asarray([-PETSYS.positionsX[X2],-PETSYS.positionsY[Y2],0])

                    
                    p1 = PETSYS.getGlobalPreRotation(event1_positions, rsectors[rSector1])
                    p2 = PETSYS.getGlobalPreRotation(event2_positions, rsectors[rSector2])
                    

                    
                    g1 = PETSYS.getGlobalRotation(p1)
                    g2 = PETSYS.getGlobalRotation(p2)




                    distance , angle = PETSYS.global_to_sinogram(g1, g2,"y")
                    slice_z=int(PETSYS.get_slice_seg((Y1,Y2)))
                    df['angle'].append(angle)
                    df['distance'].append(distance)
                    df["slice"].append(slice_z)

                    print(f"distance {distance} , angle {angle}")
                    PETSYS.fill_sino(distance,angle,slice_z)

                    sinogram_np = PETSYS.get_sinogram()

                    
                    plt.title("Sinograma Numpy")
                    plt.imshow(sinogram_np[7,:,:])

                    plt.show()



            elif detector_config["process_as"] =="GateGlobalPositions":
                """
                Here the processing is made only using global position from each event
                """    

                if runID == previous:
                    
                    #PETSYS.rotateBy(id*15)
                    
                    #get global positions (here it doesnt have to calculate because
                    #gate already gives it in global position therms)
                    event1_positions = np.asarray([globalPosX1,globalPosY1,globalPosZ1])      
                    event2_positions = np.asarray([globalPosX2,globalPosY2,globalPosZ2])



                    #g1 = np.asarray([globalPosX1,globalPosY1,globalPosZ1])      
                    #g2 = np.asarray([globalPosX2,globalPosY2,globalPosZ2])

          
                    #as our detectors rotate, we apply a rotation from each event, to
                    #represent the source rotation.
                    g1 = PETSYS.getGlobalRotation(event1_positions)
                    g2 = PETSYS.getGlobalRotation(event2_positions)

                    #gets the current angles .... but it will be used?
                    current_angle = PETSYS.getAngleOfRotation()%180

                    #uses global to sinogram to transform the positions g1 and g2 into a LOR
                    distance,angle=PETSYS.global_to_sinogram(g1,g2,"y")



                    slice_z=int(PETSYS.get_slice_mono((g1[1],g2[1])))
                    
                    df['angle'].append(angle)
                    df['distance'].append(distance)
                    df["slice"].append(slice_z)

                    PETSYS.fill_sino(distance,angle,slice_z)
                else:

                    cum += angle_step
                    previous = runID

                    # ESTE CARA FAZ O GANTRY ROTACIONAR

                    PETSYS.rotateGantry(-angle_step)
                    current_angle = PETSYS.getAngleOfRotation()%180

                    event1_positions = np.asarray([globalPosX1,globalPosY1,globalPosZ1])      
                    event2_positions = np.asarray([globalPosX2,globalPosY2,globalPosZ2])                      
                    g1 = PETSYS.getGlobalRotation(event1_positions)
                    g2 = PETSYS.getGlobalRotation(event2_positions)
                    
                    print("current angle ",current_angle)


                    current_angle = PETSYS.getAngleOfRotation()
                    distance,angle=PETSYS.global_to_sinogram(g1,g2,"y")


                    slice_z=int(PETSYS.get_slice_mono((g1[1],g2[1])))
                    df['angle'].append(angle)
                    df['distance'].append(distance)
                    df["slice"].append(slice_z)
                    PETSYS.fill_sino(distance,angle,slice_z)
                    sinogram_np = PETSYS.get_sinogram()
                    
                    # plt.title("Sinograma Numpy")
                    # plt.imshow(sinogram_np[7,:,:])
                    # plt.show()

                    #SE VOCE DESCOMENTAR ELE DEIXA DE ATUALIZAR O SINOGRAMA
                    
            
            elif detector_config["process_as"] == "PaulGlobalPosition":              

                    #PETSYS.rotateBy(id*15)
                    event1_positions = np.asarray([globalPosX1,globalPosY1,globalPosZ1])      
                    event2_positions = np.asarray([globalPosX2,globalPosY2,globalPosZ2])
                    #g1 = np.asarray([globalPosX1,globalPosY1,globalPosZ1])      
                    #g2 = np.asarray([globalPosX2,globalPosY2,globalPosZ2])

          

                    g1 = PETSYS.getGlobalRotation(event1_positions)
                    g2 = PETSYS.getGlobalRotation(event2_positions)

                    current_angle = PETSYS.getAngleOfRotation()%180

                    distance,angle=PETSYS.global_to_sinogram(g1,g2,"z")


                    slice_z=int(PETSYS.get_slice_mono((g1[2],g2[2])))
 

                    # slice_z=int(PETSYS.get_slice_mono((globalPosZ1,globalPosZ2)))
                    df['angle'].append(angle)
                    df['distance'].append(distance)
                    df["slice"].append(slice_z)
                    print("--------------------before fill sino ------------------------------")
                    print(f" g1 {g1}")
                    print(f" g2 {g2}")
                    print(f"distance {distance} , angle {angle}, slice_z {slice_z}")
                    print("-------------------------------------------------------------------")

                    PETSYS.fill_sino(distance,angle,slice_z)
               
            else:
                print( "wrong configuration parameter on 'process_as', possibles are GateGlobalPositions and CrystalID")

        
        
        elif detector_config["data_from"] == "PETSYS":
            #TODO: configure for petsys will be the same for Segmented
            crystalID1 = coincidences['crystalID1'][event] 
            crystalID2 = coincidences['crystalID2'][event]
            globalPosX1= coincidences["globalPosX1"][event]
            globalPosY1= coincidences["globalPosY1"][event]
            globalPosZ1= coincidences["globalPosZ1"][event]
            globalPosX2= coincidences["globalPosX2"][event]
            globalPosY2= coincidences["globalPosY2"][event]
            globalPosZ2= coincidences["globalPosZ2"][event]
            rSector1= coincidences["rsectorID1"][event]
            rSector2= coincidences["rsectorID2"][event]
            runID = coincidences["runID"][event]
    
    df = pd.DataFrame(df)
    print(df['distance'].unique())
    sin = PETSYS.get_sinogram()
    return sin,df
    