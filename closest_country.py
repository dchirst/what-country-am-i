import time

from shapely.errors import TopologicalError
from typing import Union
import geopandas as gpd
import numpy as np
from shapely.affinity import scale, rotate, translate
from shapely.geometry import Polygon, MultiPolygon
from hashlib import sha1
from streamlit import progress


def load_countries() -> gpd.GeoDataFrame:
    """
    Load countries from file

    :return: List of countries, reprojected to azimuthal projection and normalised to be between [-1, 1]
    :rtype: gpd.GeoDataFrame
    """
    return gpd.read_file("countries_simplified.geojson")


def accuracy_function(im: Polygon, country: Union[Polygon, MultiPolygon]) -> float:
    """
    accuracy function for image/country comparison.

    :param im: Image polygon to compare with country
    :type im: Polygon
    :param country: Country polygon to compare with image
    :type country: Union[Polygon, MultiPolygon]
    :return: Similarity index (higher is more similar) between 0 and 1
    :rtype: float
    """
    return im.intersection(country).area/max(im.area, country.area)


def accuracy(im, country, poly_map, scale_val, angle, xoff, yoff) -> float:
    """
    Given a set of transformations for the image, calculates the accuracy between the image polygon and the country polygon

    :param im: Image polygon to compare with country
    :param country: Country polygon to compare with image
    :param scale_val: scale factor for image
    :param angle: Angle by which image is rotated
    :param xoff: Translation in the x-axis by which image is moved
    :param yoff: Translation in the y-axis by which image is moved
    :return: Similarity between 0 and 1 (higher is more similar)
    """
    h = hash(f"{scale_val}{angle}{xoff}{yoff}")
    if h in poly_map:
        im_translated = poly_map[h]
    else:
        im_translated = scale(rotate(translate(im, xoff=xoff, yoff=yoff), angle=angle), xfact=scale_val, yfact=scale_val)
        poly_map[h] = im_translated
    return accuracy_function(im_translated, country)


def gradient_descent(country: Union[Polygon, MultiPolygon],
                     im: Union[Polygon, MultiPolygon],
                     iterations: int = 100,
                     starting_angle: int = 0,
                     poly_map: dict = None) -> tuple:
    """
    Perform gradient descent to find closest intersection between a single country and the given image polygon

    :param country: Normalised and reprojected country polygon
    :type country: Union[Polygon, MultiPolygon]
    :param im: Normalised polygon of the shape of the input image
    :type im: Union[Polygon, MultiPolygon]
    :param iterations: (optional) Number of iterations to perform gradient descent (default 100)
    :type iterations: int
    :param starting_angle: (optional) Starting angle in degrees for the input image. Perform the same gradient descent at multiple angles to avoid local maxima (default 0)
    :type starting_angle: int
    :param poly_map: (optional) A hash map containing transformed image polygons, to avoid having to reperformed the transformation
    :type poly_map: dict
    :return: A tuple containing the optimal theta values (scale value, angle, x offset, y offset) and the maximum accuracy found
    :rtype: tuple
    """
    if not poly_map:
        poly_map = {}

    # initialise starting theta values
    p = np.array([1, starting_angle, 0, 0])
    # Set increment values
    inc = np.array([0.001, 1, 0.001, 0.001])
    prev_accuracy = accuracy(im, country, poly_map, *p)
    accuracy_map = {}
    # Iterate
    for it in range(iterations):
        dp = inc
        # Iterate through the theta values
        for i in range(len(dp)):
            test_p = p
            # Test an updated theta value to see if it improves the accuracy
            test_p[i] = p[i] + inc[i]
            # calculate accuracy from hash map or from computation
            h = sha1(p)
            if h in accuracy_map:
                accuracy_value = accuracy_map[h]
            else:
                accuracy_value = accuracy(im, country, poly_map, *test_p)
                accuracy_map[h] = accuracy_value

            # Update the theta value positively if it increased the accuracy, negatively if it decreased the accuracy
            if accuracy_value > prev_accuracy:
                dp[i] = inc[i]
                prev_accuracy = accuracy_value
            else:
                dp[i] = -inc[i]
        p = p + dp

    return p, prev_accuracy


def find_closest_country(polygon: Polygon, _pbar: progress = None) -> tuple:
    """
    Finds the closest country for a given image polygon

    :param polygon: Shapely polygon that you'd like to compare to countries
    :param _pbar: (optional) Streamlit progress bar to show progress of the algorithm
    :return: A tuple containing a GeoDataFrame with only the most accurate country, the best accuracy score for that
    country, and the corresponding scale factor, angle, x offset and y offset that transformed the polygon to generate
    that accuracy score
    :rtype: tuple
    """

    start_time = time.time()

    countries = load_countries()

    best_country = best_accuracy = best_theta = 0
    length = countries.shape[0]
    poly_map = {}

    for idx, country in countries.iterrows():
        for a in (0, 90, 180, 270):
            # TODO: Fix Fiji and Russia for which there are topological errors
            try:
                theta, c_accuracy = gradient_descent(country.geometry, polygon, iterations=100, starting_angle=a, poly_map=poly_map)
            except TopologicalError:
                print(f"TOPOLOGICAL ERROR for {country['ADMIN']}")
                continue

            if c_accuracy > best_accuracy:
                best_country = country["ADMIN"]
                best_accuracy = c_accuracy
                best_theta = theta
                print("New best country:", best_country, best_accuracy, theta)

            # break if the best accuracy for a given angle is so low that it will likely never get above the current
            # best accuracy
            if best_accuracy - c_accuracy > 0.5:
                print("breaking")
                break
            print(f"{country['ADMIN']}: {c_accuracy}")

        if _pbar:
            _pbar.progress(idx / length)

    print("Time taken:", time.time() - start_time)
    return countries[countries["ADMIN"] == best_country], best_accuracy, best_theta

