import numpy as np
import platform
from GimnTools.ImaGIMN.IO.MIIO import MIIO

class image(MIIO):
    """
    @class image
    @brief A class to handle image reading and properties.

    The `image` class implements methods to read images from files or numpy
    arrays, manage image metadata, and provide access to image pixel data.
    """

    __np_pixels = None
    __name = None
    __spacing = None
    __origin = None
    __size = None
    __image_path = None
    __is_empty = True
    __out_path = ""

    def __init__(self, image=None, is_serie=False):
        """
        @brief Initializes the image class.

        The constructor accepts either a numpy array or a string representing
        the image path. If the input is a string and `is_serie` is set to
        True, it reads a series of images; otherwise, it reads a single image.

        @param image A numpy array representing the image or a string path to 
                     the image file.
        @param is_serie A boolean indicating if the input is a series of images.
        """
        if isinstance(image, np.ndarray):
            self.set_image(image)
        elif isinstance(image, str):
            if is_serie:
                self.read_serie(image)
            else:
                self.read_image(image)

    def set_image(self, image):
        """
        @brief Sets the image property as a numpy array.

        This method sets the internal pixel representation of the image.

        @param image A numpy array representing the image.
        """
        self.__np_pixels = np.asarray(image)
        self.__is_empty = False

    def set_name(self, name):
        """
        @brief Sets the name of the image.

        The name is used to save the image on the PC.

        @param name A string representing the name of the image.
        """
        self.__name = name

    @property
    def pixels(self):
        """
        @brief Gets the pixel data of the image.

        @return A numpy array containing the pixel data of the image.
        """
        return self.__np_pixels


    @property
    def is_empty(self):
        """
        @brief Checks if the image is empty.

        @return A boolean indicating if the image is empty.
        """
        return self.__is_empty

    @property
    def name(self):
        """
        @brief Gets the name of the image.

        @return A string representing the name of the image.
        """
        return self.__name

    @property
    def path(self):
        """
        @brief Gets the path of the image.

        @return A string representing the path of the image.
        """
        return self.__image_path

    def set_out_path(self, out_path):
        """
        @brief Sets the output path for saving the image.

        This method ensures that the output path ends with the appropriate
        directory separator based on the operating system.

        @param out_path A string representing the output path where the image 
                         will be saved.
        """
        self.__out_path = out_path
        end = "/" if platform.system() == "Linux" else "\\"
        if self.__out_path[-1] != end:
            self.__out_path += end

    @property
    def out_path(self):
        """
        @brief Gets the output path for saving the image.

        @return A string representing the output path for saving the image.
        """
        return self.__out_path

    @path.setter
    def path(self, value):
        """
        @brief Sets the image path.

        This setter allows setting the image path, ensuring it is a string.

        @param value A string representing the image path.
        """
        if isinstance(value, str):
            self.__image_path = value
        else:
            print("path is not a string")
