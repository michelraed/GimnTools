from GimnTools.ImaGIMN.processing.filters.spatial_filters import get_neighbors
import numpy as np

def doErosion(img,neighboors):
  """
    @brief Applies morphological erosion to a grayscale image.
   
    This function performs the erosion operation on the input image using a specified neighborhood size.
    For each pixel, it replaces the pixel value with the minimum value found in its neighborhood.
   
    @param img numpy.ndarray
      Input 2D grayscale image to be eroded.
    @param neighboors int
      Size of the neighborhood to use for erosion (e.g., 3 for a 3x3 neighborhood).
   
    @return numpy.ndarray
      The eroded image as a 2D numpy array with the same shape as the input.
   
    @note
      The function assumes that the helper function `get_neighbors(img, x, y, size)` is defined elsewhere,
      and that it returns a neighborhood of the specified size centered at (x, y).

  """
  nx = img.shape[1]
  ny = img.shape[0]
  out =  np.zeros(img.shape)

  for x in range(nx):
    for y in range(ny):
      neighbors = get_neighbors (img,x,y,neighboors )
      min = neighbors.min()
      out[y,x]=min
  return out


def doDilation( img,neighboors):
  """
  @brief Applies a morphological dilation filter to a grayscale image.


  @param img (numpy.ndarray): Input 2D grayscale image as a NumPy array.
  @param neighboors (int or iterable): Defines the neighborhood structure for dilation. 

  @return numpy.ndarray: The dilated image as a NumPy array with the same shape as the input.

  @note The function assumes that the `get_neighbors` function is defined elsewhere and 
  """
  nx = img.shape[1]
  ny = img.shape[0]
  out =  np.zeros(img.shape)

  for x in range(nx):
    for y in range(ny):
      neighbors = get_neighbors (img,x,y,neighboors )
      max = neighbors.max()
      out[y,x]=max
  return out

def doOpen(img,neighboors):
  """
  @brief Performs morphological opening on an image.

  This function applies the morphological opening operation to the input image.
  Morphological opening is defined as an erosion followed by a dilation, using the same structuring element (neighboors).
  It is typically used to remove small objects from an image while preserving the shape and size of larger objects.

  @param img Input image to be processed.
  @param neighboors Structuring element or neighborhood definition used for erosion and dilation.

  @return The image after morphological opening.
  """
  out1= doErosion(img,neighboors)
  out= doDilation(out1,neighboors)
  return out
  
def doClose( img,neighboors):      
  """
    @brief Performs morphological closing on an image.

    This function applies the morphological closing operation, which consists of a dilation
    followed by an erosion, using the specified neighborhood structure.

    @param img Input image to be processed.
    @param neighboors Structuring element or neighborhood definition used for the morphological operations.

    @return The image after applying the closing operation.
  """
  out1= doDilation(img,neighboors)
  out= doErosion(out1,neighboors)
  return out
