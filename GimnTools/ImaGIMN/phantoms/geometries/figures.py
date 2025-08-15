from PIL import Image, ImageDraw
import math
import numpy as np

def create_star_image(size, num_points, x, y, star_radius,fill=1):
    """
    /**
     * @brief Creates a grayscale image with a star-shaped polygon.
     *
     * This function generates a square grayscale image of the specified size, draws a star with a given number of points,
     * centered at (x, y), and with a specified outer radius. The star is filled with the provided gray level value.
     *
     * @param size The width and height of the output image (in pixels).
     * @param num_points The number of points (arms) of the star.
     * @param x The x-coordinate of the center of the star.
     * @param y The y-coordinate of the center of the star.
     * @param star_radius The outer radius of the star (distance from center to outermost point).
     * @param fill The gray level value to fill the star (default is 1).
     * @return A NumPy array representing the generated grayscale image with the star.
     */
    """
    # Create a new grayscale image
    data = np.zeros([size, size], dtype=np.int64)
    image = Image.fromarray(data, mode='I;16')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Calculate the radius and center of the star
    radius = size // 2
    center = (x, y)

    # Calculate the points of the star
    points = []
    for i in range(num_points * 2):
        angle = i * math.pi / num_points
        r = star_radius if i % 2 == 0 else star_radius / 2  # alternate between outer and inner radius
        x_point = center[0] + r * math.cos(angle)
        y_point = center[1] + r * math.sin(angle)
        points.append((x_point, y_point))

    # Draw the star
    draw.polygon(points, fill=fill)  # 10 represents a gray level

    image=np.asarray(image)
    # Return the image

    return image



def create_ellipse_image(size, x, y, width, height,fill):
    """
        /**
         * @brief Creates a grayscale image with a filled ellipse.
         *
         * This function generates a 2D numpy array representing a grayscale image of the specified size,
         * with a filled ellipse drawn at the given position and dimensions.
         *
         * @param size The width and height of the square image (in pixels).
         * @param x The x-coordinate of the center of the ellipse.
         * @param y The y-coordinate of the center of the ellipse.
         * @param width The width of the ellipse (in pixels).
         * @param height The height of the ellipse (in pixels).
         * @param fill The grayscale fill value for the ellipse (e.g., 0 for black).
         * @return numpy.ndarray The resulting image as a 2D numpy array.
         */
    """
    # Create a new grayscale image
    #image = Image.new('L', (size, size), 255)  # 255 representa branco

    data = np.zeros([size, size], dtype=np.int64)
    image = Image.fromarray(data, mode='I;16')
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define as coordenadas da elipse
    left = x - width // 2
    top = y - height // 2
    right = x + width // 2
    bottom = y + height // 2

    # Desenhe a elipse na imagem
    draw.ellipse((left, top, right, bottom), fill=fill)  # 0 representa preto
    image=np.asarray(image)
    # Retorne a imagem
    return image


def create_polygon_image(size, num_sides, x, y, radius,fill=1):
    """
        /**
         * @brief Creates a grayscale image with a filled regular polygon.
         *
         * This function generates a 2D NumPy array representing a grayscale image of the specified size,
         * with a regular polygon (with a given number of sides, center, and radius) drawn and filled with the specified value.
         *
         * @param size The width and height of the square image (in pixels).
         * @param num_sides The number of sides of the regular polygon.
         * @param x The x-coordinate of the center of the polygon.
         * @param y The y-coordinate of the center of the polygon.
         * @param radius The radius of the polygon (distance from center to vertex).
         * @param fill The fill value for the polygon (default is 1).
         * @return np.ndarray The resulting image as a 2D NumPy array.
         */
    """
    # Create a new grayscale image
    #image = Image.new('L', (size, size), 255)  # 255 representa branco
    data = np.zeros([size, size], dtype=np.int64)
    image = Image.fromarray(data, mode='I;16')
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Calculate the points of the polygon
    points = []
    for i in range(num_sides):
        angle = i * 2 * math.pi / num_sides
        x_point = x + radius * math.cos(angle)
        y_point = y + radius * math.sin(angle)
        points.append((x_point, y_point))

    # Draw the polygon on the image
    draw.polygon(points, fill=fill)  # 0 representa preto
    image=np.asarray(image)
    # Return the image
    return image


def create_circle_image(size, x, y, radius,fill):
    """
        /**
         * @brief Creates a grayscale image with a filled circle.
         *
         * This function generates a square grayscale image of the specified size,
         * draws a filled circle at the given (x, y) coordinates with the specified radius and fill value,
         * and returns the resulting image as a NumPy array.
         *
         * @param size The width and height of the square image (in pixels).
         * @param x The x-coordinate of the center of the circle.
         * @param y The y-coordinate of the center of the circle.
         * @param radius The radius of the circle (in pixels).
         * @param fill The grayscale fill value for the circle (e.g., 0 for black).
         * @return NumPy array representing the generated image with the drawn circle.
         */
    """
    # Create a new grayscale image
    #image = Image.new('L', (size, size), 255)  # 255 representa branco
    data = np.zeros([size, size], dtype=np.int64)
    image = Image.fromarray(data, mode='I;16')
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define as coordenadas do círculo
    left = x - radius
    top = y - radius
    right = x + radius
    bottom = y + radius

    # Desenhe o círculo na imagem
    draw.ellipse((left, top, right, bottom), fill=fill)  # 0 representa preto
    image=np.asarray(image)

    # Retorne a imagem
    return image
