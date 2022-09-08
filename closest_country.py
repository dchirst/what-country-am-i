from typing import Union

import geopandas as gpd
import numpy as np
import psycopg2
import pyproj
import streamlit as st
from shapely.affinity import scale, rotate, translate
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon


# # Initialize connection.
# # Uses st.experimental_singleton to only run once.
# @st.experimental_singleton
# def init_connection():
#     return psycopg2.connect(**st.secrets["postgres"])
#
#
# conn = init_connection()
#
# # Perform query.
# # Uses st.experimental_memo to only rerun when the query changes or after 10 min.
# @st.experimental_memo(ttl=600)
# def run_query(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         return cur.fetchall()


def load_countries():
    return gpd.read_file("countries.geojson")
    # return gpd.GeoDataFrame.from_postgis("SELECT name, geometry FROM countries;",
    #                                      conn,
    #                                      geom_col="geometry")


def cost_function(im: Polygon, country: Union[Polygon, MultiPolygon]) -> float:
    """
    Cost function for image/country comparison.

    :param im: Image polygon to compare with country
    :type im: Polygon
    :param country: Country polygon to compare with image
    :type country: Union[Polygon, MultiPolygon]
    :return: Similarity index (higher is more similar) between 0 and 1
    :rtype: float
    """
    return im.intersection(country).area/max(im.area, country.area)


def cost(im, country, scale_val, angle, xoff, yoff) -> float:
    """
    Given a set of transformations for the image, calculates the cost between the image polygon and the country polygon
    :param im: Image polygon to compare with country
    :param country: Country polygon to compare with image
    :param scale_val: scale factor for image
    :param angle: Angle by which image is rotated
    :param xoff: Translation in the x-axis by which image is moved
    :param yoff: Translation in the y-axis by which image is moved
    :return: Similarity between 0 and 1 (higher is more similar)
    """
    im_translated = scale(rotate(translate(im, xoff=xoff, yoff=yoff), angle=angle), xfact=scale_val, yfact=scale_val)
    return cost_function(im_translated, country)


def gradient_descent(country, im, iterations=100, starting_angle=0):
    p = np.array([1, starting_angle, 0, 0])
    inc = np.array([0.001, 1, 0.001, 0.001])
    prev_cost = cost(im, country, *p)
    p_list = []
    for it in range(iterations):
        dp = inc
        cost_iterations = []
        for i in range(len(dp)):
            test_p = p
            test_p[i] = p[i] + inc[i]
            cost_value = cost(im, country, *test_p)
            cost_iterations.append(cost_value)
            if cost_value > prev_cost:
                dp[i] = inc[i]
                prev_cost = cost_value
            else:
                dp[i] = -inc[i]

        # print(it, max(cost_iterations))

        p = p + dp
        if it % 10 == 0:
            p_list.append(p)

    return p_list, prev_cost


@st.experimental_memo(ttl=600)
def find_closest_country(polygon, _pbar=None):
    wgs84 = pyproj.CRS('EPSG:4326')

    countries = load_countries()
    max_cost = 0
    best_country = best_theta = best_cost = 0
    length = countries.shape[0]
    print(countries)
    print(polygon)
    for idx, country in countries.iterrows():
        # country_reprojected = gpd.GeoSeries(country).to_crs(azimuthal_projection)
        for a in (0, 90, 180, 270):
            theta, c_cost = gradient_descent(country.geometry, polygon, iterations=100, starting_angle=a)

            if c_cost > max_cost:
                max_cost = c_cost
                best_country = country["name"]
                best_theta = theta
                best_cost = c_cost
                print("New best country:", best_country, best_cost)

            print(f"{country['name']}: {best_cost}")
            if _pbar:
                _pbar.progress(idx / length)

    return countries[countries["name"] == best_country], best_cost


def make_country_plot(country: gpd.GeoSeries):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.set_facecolor('blue')
    plt.axis('off')
    country.plot(ax=ax, color="#0fd159")
    return fig


if __name__ == '__main__':
    c = load_countries()

    c.to_file("countries.geojson")

