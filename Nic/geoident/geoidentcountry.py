import os
import time
import geopandas as gpd
from shapely.geometry import Point

country_json_file = 'shapefiles\\country_json\\world-administrative-boundaries.geojson'

# Load the GeoJSON file and create a spatial index
script_dir = os.path.dirname(os.path.realpath(__file__))
shp_file_path = os.path.join(script_dir, country_json_file)
world = gpd.read_file(shp_file_path)
sindex = world.sindex

def is_country(coords) -> dict:
    # Create a point from the coordinates
    point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

    # Use the spatial index to find the rows that contain the point
    possible_matches_index = list(sindex.intersection(point.bounds))
    possible_matches = world.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(point)]

    for _, row in precise_matches.iterrows():
        return {'country': row['name'], 'region': row['region'], 'country_code': row['iso3']}

    return {'country': 'No country found', 'region': 'No region found', 'country_code': 'No country code found'}


start_time = time.time()
print(is_country((47.3779506,8.534353)))
end_time = time.time()
execution_time = end_time - start_time
print(f"The function took {execution_time} seconds to execute.")