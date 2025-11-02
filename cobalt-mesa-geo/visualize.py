# visualize.py

import folium
import geopandas as gpd

def plot_snapshot(geojson_path="logs/snapshot_agents.geojson", provinces="data/provinces.geojson", out_html="logs/map.html"):
    gdf_agents = gpd.read_file(geojson_path)
    gdf_prov = gpd.read_file(provinces)

    # center map on agents
    cent = gdf_agents.geometry.unary_union.centroid
    m = folium.Map(location=[cent.y, cent.x], zoom_start=5)

    # provinces layer
    folium.GeoJson(gdf_prov).add_to(m)

    # agent points with popup
    for _, row in gdf_agents.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        popup = folium.Popup(f"{row['id']}<br>type: {row['type']}<br>cash: {row['cash']}<br>inv: {row['inventory']}", max_width=300)
        folium.CircleMarker(location=[lat, lon], radius=6, popup=popup).add_to(m)

    m.save(out_html)
    print(f"Map saved to {out_html}")
