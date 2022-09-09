import psycopg2
import streamlit as st
import csv
import geopandas as gpd
from shapely.affinity import scale, translate
from sqlalchemy import text, create_engine
from sqlalchemy.future import engine


# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

# Perform query.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def make_country_unit(poly):
    x, y = poly.centroid.coords[0]
    minx, miny, maxx, maxy = poly.bounds
    fact = 1 / max(x-minx, maxx-x, maxy-y, y-miny)
    unit_poly = scale(poly, xfact=fact, yfact=fact)

    x_translation, y_translation = unit_poly.centroid.coords[0]
    final_poly = translate(unit_poly, xoff=-x_translation, yoff=-y_translation)
    return final_poly


if __name__ == '__main__':
    # engine = create_engine('postgresql://dhirst:postgis@localhost/postgis')
    # print(engine.connect())
    #
    # gdf = gpd.GeoDataFrame.from_postgis("SELECT admin, wkb_geometry FROM destination_table;", conn, geom_col="wkb_geometry")
    # gdf.rename(columns={
    #     "admin": "name",
    #     "wkb_geometry": "geometry"
    # }, inplace=True)
    # gdf["geometry"] = gdf.geometry.apply(make_country_unit)
    # with engine.connect() as connection:
    #     gdf.to_postgis(name="countries", con=connection)
    wgs84 = pyproj.CRS
    df = gpd.read_file("/Users/dhirst/Downloads/countries.geojson")
    print(df)