import os
import time
import geopandas as gpd
from shapely.geometry import Point

country_json_file = 'shapefiles\\urban_shp\\ne_50m_urban_areas.shp'

# Load the GeoJSON file and create a spatial index
script_dir = os.path.dirname(os.path.realpath(__file__))
shp_file_path = os.path.join(script_dir, country_json_file)
urban_areas = gpd.read_file(shp_file_path)
sindex = urban_areas.sindex

def is_urban(coords):
    # Create a point from the coordinates
    point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

    # Use the spatial index to find the rows that contain the point
    possible_matches_index = list(sindex.intersection(point.bounds))
    possible_matches = urban_areas.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(point)]

    # If any matches are found, return True
    if not precise_matches.empty:
        return True
    return False

coordinates = [
    (47.3779506,8.534353),  # Zurich, Switzerland
    (34.052235, -118.243683),  # Los Angeles, USA
    (-33.865143, 151.209900),  # Sydney, Australia
    (55.755826, 37.617600),  # Moscow, Russia
    (-22.906847, -43.172897),  # Rio de Janeiro, Brazil
    (35.689487, 139.691711),  # Tokyo, Japan
    (28.613939, 77.209023),  # New Delhi, India
    (-1.286389, 36.817223),  # Nairobi, Kenya
    (64.126520, -21.817439),  # Reykjavik, Iceland
    (-25.263740, -57.575926),  # Asuncion, Paraguay
    (32.9681685240936, -120.75474968007332) # Ocean
]
start_time = time.time()
for coord in coordinates:
    print(is_urban(coord))
end_time = time.time()
execution_time = end_time - start_time
print(f"The function took {execution_time} seconds to execute.")