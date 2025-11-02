# data_gen.py
# makes a DRC sample data with a few mines, a refinery, and province polygos for a demo.

import json, os
from shapely.geometry import Point, mapping, Polygon
import geopandas as gpd
import pandas as pd

os.makedirs("data", exist_ok=True)

# sample mines (lon, lat roughly in central Africa for demo)
mines = [
    {"id": "mine_1", "name": "Kivu ASM", "type": "artisanal", "lon": 27.5, "lat": -2.5, "prod": 10},
    {"id": "mine_2", "name": "Katanga Major", "type": "industrial", "lon": 25.0, "lat": -10.5, "prod": 300},
    {"id": "mine_3", "name": "Ituri ASM", "type": "artisanal", "lon": 29.0, "lat": 1.5, "prod": 15}
]
mines_df = pd.DataFrame(mines)
mines_gdf = gpd.GeoDataFrame(
    mines_df,
    geometry=[Point(xy) for xy in zip(mines_df.lon, mines_df.lat)],
    crs="EPSG:4326"
)
mines_gdf.to_file("data/cobalt_mines.geojson", driver="GeoJSON")

# sample refineries
refineries = [
    {"id": "ref_1", "name": "Lubumbashi Refinery", "lon": 25.5, "lat": -11.7},
    {"id": "ref_2", "name": "Coast Port Refinery", "lon": 48.5, "lat": -4.0}  # pretend foreign/port
]
ref_df = pd.DataFrame(refineries)
ref_gdf = gpd.GeoDataFrame(ref_df, geometry=[Point(xy) for xy in zip(ref_df.lon, ref_df.lat)], crs="EPSG:4326")
ref_gdf.to_file("data/refineries.geojson", driver="GeoJSON")

# simple province polygons (rectangles for demo)
provinces = [
    {"name": "Province A", "geometry": Polygon([(24,-12),(31,-12),(31,-6),(24,-6)])},
    {"name": "Province B", "geometry": Polygon([(24,-6),(31,-6),(31,2),(24,2)])}
]
prov_gdf = gpd.GeoDataFrame(provinces, crs="EPSG:4326")
prov_gdf.to_file("data/provinces.geojson", driver="GeoJSON")

print("Sample data generated in ./data/")