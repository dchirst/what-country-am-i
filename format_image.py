import shapely
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import cv2
from shapely.affinity import scale, translate
from rembg import remove
import matplotlib.pyplot as plt
from streamlit.runtime.uploaded_file_manager import UploadedFile


def file_upload_to_array(image_stream: UploadedFile) -> np.ndarray:
    """
    Convert uploaded streamlit image stream to a numpy array

    :param image_stream: uploaded image from Streamlit
    :type image_stream: UploadedFile
    :return: Numpy array of the image
    :rtype: np.ndarray
    """
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def format_image(file: UploadedFile) -> np.ndarray:
    """
    Read uploaded image from streamlit, remove the background, and return array

    :param file: Uploaded image from Streamlit
    :type file: UploadedFile
    :return: Numpy array with background of image removed
    :rtype: np.ndarray
    """
    img = file_upload_to_array(file)
    background_removed_img = remove(img)
    return background_removed_img[..., [2, 1, 0]]


def im_to_polygon(im: np.ndarray) -> shapely.geometry.Polygon:
    """
    Convert an image without a background into a Shapely Polygon
    :param im: Numpy array representing an RGB image with background removed
    :type im: np.ndarray
    :return: Shapely Polygon showing the exterior of an image. Has been normalised to be between -1 and 1 around the origin
    :rtype: shapely.geometry.Polygon
    """

    # Find contours of image, showing outline
    mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Convert contours to multipolygon
    contours = map(np.squeeze, contours)  # removing redundant dimensions
    filtered_contours = filter(lambda x: len(x) > 2, contours)
    polygons = map(Polygon, filtered_contours)  # converting to Polygons
    multipolygon = MultiPolygon(polygons)

    # only return the biggest polygon
    polygon = max(multipolygon, key=lambda a: a.area)

    # Normalise between -1 and 1 around origin
    x, y = polygon.centroid.coords[0]
    minx, miny, maxx, maxy = polygon.bounds
    fact = 1 / max(x-minx, maxx-x, maxy-y, y-miny)
    unit_poly = scale(polygon, xfact=-fact, yfact=-fact)
    x_translation, y_translation = unit_poly.centroid.coords[0]
    final_poly = translate(unit_poly, xoff=-x_translation, yoff=-y_translation)

    return final_poly.simplify(0.02)



