import SimpleITK as sitk
import numpy as np
import itk
import time
from GimnTools.ImaGIMN.IO.GimnIO import GimnIO
from copy import deepcopy
import os



class MIIO:
    """
    @class MIIO
    @brief A class for handling DICOM image input and output operations.

    The `MIIO` class provides methods to read, write, and manipulate DICOM images, 
    including normalization and metadata handling.
    """

    def __init__(self):
        """
        @brief Initializes the MIIO class.

        This constructor does not take any parameters and initializes the class.
        """
        pass

    @staticmethod
    def renormalize(image, dtype):
        """
        @brief Renormalizes the input image to the specified data type.

        This method normalizes each slice of the input image to fit within the 
        range of the specified data type.

        @param image A 3D numpy array representing the image to be normalized.
        @param dtype The desired data type to which the image should be cast.

        @return A tuple containing the normalized image and a dictionary with 
                regression parameters (slope and intercept).
        """
        image = deepcopy(image)
        slices = np.shape(image)[0]
        normalized = np.zeros_like(image).astype(dtype)
        regression = {"slope": [], "intercept": []}

        for slc in range(slices):
            min_value = image[slc].min()
            max_value = image[slc].max()
            rescale_slope = (max_value - min_value) / np.iinfo(dtype).max
            rescale_intercept = min_value

            normalized[slc] = ((image[slc] - min_value) / rescale_slope).astype(dtype)
            regression['slope'].append(float(rescale_slope))
            regression['intercept'].append(float(rescale_intercept))

        return normalized, regression

    @staticmethod
    def save_dicom(image, nome_arquivo, origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0), save_json=True):
        """
        @brief Saves a numpy array as a DICOM file.

        This method normalizes the image, sets the necessary metadata, and 
        saves the image as a DICOM file.

        @param image A 3D numpy array representing the image to save.
        @param nome_arquivo The filename (including path) to save the DICOM file.
        @param origin A tuple representing the origin of the image.
        @param spacing A tuple representing the spacing of the image.
        @param save_json A boolean indicating whether to save regression data as JSON.
        """
        # Normalize the float values to uint16 range and cast to uint16
        image, regression = MIIO.renormalize(image, np.uint16)
        image = sitk.GetImageFromArray(image)

        folder_path = GimnIO.check_and_create_folder(nome_arquivo)
        name = nome_arquivo.split(sep="/")[-1]
        nome_arquivo = folder_path + "/" + name

        print("NOME DO ARQUIVO", nome_arquivo)
        print(folder_path)

        # Define origin and spacing
        image.SetOrigin(origin)
        image.SetSpacing(spacing)

        # Fill in necessary metadata
        image.SetMetaData("0010|0010", "Nome do Paciente")  # Patient Name
        image.SetMetaData("0008|0060", "CT")  # Modality
        image.SetMetaData("0008|103e", "Descrição da Série")  # Series Description
        image.SetMetaData("0028|0004", "MONOCHROME2")  # Photometric Interpretation
        image.SetMetaData("0028|0100", str(16))  # Bits Allocated
        image.SetMetaData("0028|0101", str(16))  # Bits Stored
        image.SetMetaData("0028|0102", str(15))  # High Bit
        image.SetMetaData("0028|0103", str(0))  # Pixel Representation (0 for unsigned)

        if save_json:
            json_name = nome_arquivo + ".json"
            GimnIO.save_json(json_name, regression)

        # Save the image as a DICOM file
        writer = sitk.ImageFileWriter()
        writer.SetFileName(f"{nome_arquivo}.dcm")
        writer.Execute(image)

    @staticmethod
    def read_dicom(folder_name, use_json=False):
        """
        @brief Reads a DICOM file from the specified folder.

        This method reads a DICOM file and returns the image data along with 
        its origin and spacing.

        @param folder_name The folder containing the DICOM file.
        @param use_json A boolean indicating whether to use regression data from JSON.

        @return A tuple containing the numpy array of the image, origin, and spacing.
        """


        file_name = folder_name + "/" + folder_name.split(sep="/")[-1]
        image = sitk.ReadImage(f"{file_name}.dcm")
        np_matrix = sitk.GetArrayFromImage(image).astype(np.float32)

        if use_json:
            regression = GimnIO.read_json(f"{file_name}.json")
            slices, height, width = np_matrix.shape

            # Renormalize to the pixel value
            for slc in range(slices):
                np_matrix[slc] = np_matrix[slc] * regression['slope'][slc] + regression['intercept'][slc]

        origin = image.GetOrigin()
        spacing = image.GetSpacing()

        return np_matrix, origin, spacing

    @staticmethod
    def save_series(image, out_dir, origin=[0, 0, 0], spacing=[1, 1, 1], pixel_dtype=np.uint16, save_json=False):
        """
        @brief Saves a series of images as DICOM files.

        This method normalizes the image series and saves each slice as a 
        separate DICOM file.

        @param image A 4D numpy array representing the image series.
        @param out_dir The output directory where the DICOM files will be saved.
        @param origin A list representing the origin of the image.
        @param spacing A list representing the spacing of the image.
        @param pixel_dtype The desired data type for the pixel values.
        @param save_json A boolean indicating whether to save regression data as JSON.
        """
        out_dir = out_dir + "_series"
        GimnIO.check_and_create_folder(out_dir)

        normalized_data, regression = MIIO.renormalize(image, pixel_dtype)

        if save_json:
            file_name = out_dir.split(sep="/")[-1]
            json_name = out_dir + '/' + file_name + ".json"
            GimnIO.save_json(json_name, regression)

        new_img = sitk.GetImageFromArray(normalized_data)
        new_img.SetSpacing(spacing)
        new_img.SetOrigin(origin)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        direction = new_img.GetDirection()

        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),  # Series Instance UID
            ("0020|0037", "\\".join(map(str, direction))),  # Image Orientation
            ("0008|103e", "Created-SimpleITK")  # Series Description
        ]

        # Write slices to output directory
        for i in range(new_img.GetDepth()):
            MIIO.writeSlices(series_tag_values, new_img, out_dir, i)

    @staticmethod
    def writeSlices(series_tag_values, new_img, out_dir, i):
        """
        @brief Writes a single slice of the image to a DICOM file.

        This method sets the metadata for the slice and saves it as a DICOM file.

        @param series_tag_values A list of tuples containing the series metadata.
        @param new_img The new image object containing the slice data.
        @param out_dir The output directory where the DICOM file will be saved.
        @param i The index of the slice to be written.
        """
        image_slice = new_img[:, :, i]

        list(
            map(
                lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]),
                series_tag_values,
            )
        )

        # Slice specific tags
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0008|0060", "CT")  # Modality
        image_slice.SetMetaData("0020|0013", str(i))  # Instance Number

        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(out_dir, str(i) + ".dcm"))
        writer.Execute(image_slice)

    @staticmethod
    def read_series(path_to_folder, pixel_type, use_json=False):
        """
        @brief Reads a series of DICOM images from the specified folder.

        This method reads all DICOM files in the specified folder and 
        returns a numpy array of the image series.

        @param path_to_folder The path to the folder containing the DICOM series.
        @param pixel_type The desired pixel type for the images.
        @param use_json A boolean indicating whether to use regression data from JSON.

        @return A numpy array representing the image series.
        """
        if use_json:
            file_name = path_to_folder.split(sep="/")[-1]
            json_name = path_to_folder + '/' + file_name + ".json"
            regression = GimnIO.read_json(json_name)

        PixelType = pixel_type
        Dimension = 3
        ImageType = itk.Image[PixelType, Dimension]

        dirName = path_to_folder
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.AddSeriesRestriction("0008|0021")
        namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(dirName)

        seriesUID = namesGenerator.GetSeriesUIDs()

        if len(seriesUID) < 1:
            print("No DICOMs in: " + dirName)
            sys.exit(1)

        print("The directory: " + dirName)
        print("Contains the following DICOM Series: ")

        for uid in seriesUID:
            print(uid)

        seriesIdentifier = uid
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()
        reader.Update()

        image_itk = reader.GetOutput()
        np_array = itk.array_view_from_image(image_itk)
        slices, height, width = np_array.shape

        # Renormalize to the pixel value
        if use_json:
            try:
                for slc in range(slices):
                    np_array[slc] = np_array[slc] * regression['slope'][slc] + regression['intercept'][slc]
            except:
                print(f"Error reading json, check if the file {json_name} exists")

        return np_array