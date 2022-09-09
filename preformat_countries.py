from typing import Union

import matplotlib.pyplot as plt
from shapely.affinity import scale, translate
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd
from pyproj import Proj
import warnings


def reproject_country(country: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    centroid = country.geometry.centroid.values[0]
    lon, lat = centroid.x, centroid.y
    azimuthal_projection = Proj(proj='aeqd', datum='WGS84', lon_0=lon, lat_0=lat, units='m').crs
    country_reprojected_geom = country.to_crs(azimuthal_projection).geometry

    return country_reprojected_geom


def transform_country(country: gpd.GeoDataFrame) -> Union[Polygon, MultiPolygon]:
    x, y = country.centroid.values[0].coords[0]
    minx, miny, maxx, maxy = tuple(country.values.bounds[0])
    fact = 1 / max(x - minx, maxx - x, maxy - y, y - miny)

    unit_poly = scale(country.values[0], xfact=fact, yfact=fact)

    x_translation, y_translation = unit_poly.centroid.coords[0]
    final_poly = translate(unit_poly, xoff=-x_translation, yoff=-y_translation)

    return final_poly


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    c = gpd.read_file("/Users/dhirst/Downloads/countries.geojson")

    for i in range(c.shape[0]):
        country = c[c.index == i]

        country_reprojected_geom = reproject_country(country)

        final_poly = transform_country(country_reprojected_geom)

        final_poly_simplified = final_poly.simplify(0.001)

        # Plot reprojected, normalised country
        country_reprojected_geom.geometry = [final_poly_simplified]
        country_reprojected_geom.plot()
        plt.title(f"{country.head().ADMIN.values[0]}")
        plt.show()

        keep = input("keep?: ")
        if keep == "y":
            c[c.index == i] = country_reprojected_geom
            print(c[c.index == i])
        else:
            c.drop(index=i, inplace=True)
            print(c.shape)

        plt.close()

    c.to_file("countries_simplified.geojson")

