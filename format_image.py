from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import cv2
from shapely.affinity import scale, translate
from rembg import remove
import matplotlib.pyplot as plt


def file_upload_to_array(image_stream) -> np.ndarray:
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def format_image(file) -> np.ndarray:
    img = file_upload_to_array(file)
    background_removed_img = remove(img)
    plt.imshow(background_removed_img)
    plt.show()
    return background_removed_img[..., [2, 1, 0]]


def im_to_polygon(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(mask.shape)
    for contour in contours:
        cv2.drawContours(mask, contour, -1, (255, 0, 0), 3)
    fig = plt.figure()
    plt.imshow(mask)
    fig.savefig("test2.png")
    contours = map(np.squeeze, contours)  # removing redundant dimensions
    filtered_contours = filter(lambda x: len(x) > 2, contours)
    polygons = map(Polygon, filtered_contours)  # converting to Polygons
    multipolygon = MultiPolygon(polygons)
    polygon = max(multipolygon, key=lambda a: a.area)
    x, y = polygon.centroid.coords[0]
    minx, miny, maxx, maxy = polygon.bounds
    fact = 1 / max(x-minx, maxx-x, maxy-y, y-miny)
    unit_poly = scale(polygon, xfact=-fact, yfact=-fact)

    x_translation, y_translation = unit_poly.centroid.coords[0]
    final_poly = translate(unit_poly, xoff=-x_translation, yoff=-y_translation)
    return final_poly.simplify(0.02)



