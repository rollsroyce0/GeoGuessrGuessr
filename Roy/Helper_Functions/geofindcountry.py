import os
import time
import geopandas as gpd
from shapely.geometry import Point
import random

country_json_file = 'shapefiles\\country_json\\world-administrative-boundaries.geojson'

# Load the GeoJSON file and create a spatial index
script_dir = os.path.dirname(os.path.realpath(__file__))
shp_file_path = os.path.join(script_dir, country_json_file)
world = gpd.read_file(shp_file_path)
sindex = world.sindex

def generate_random_country_code(continent=None):   
    
    if continent is not None:
        continent = continent.lower()
        continent_codes = world[world['continent'].str.lower() == continent].iso3.unique()
        continent_codes = [code for code in continent_codes if code is not None]
        return random.choice(continent_codes)
    else:
        codes = world.iso3.unique()
        codes = [code for code in codes if code is not None]
        return random.choice(codes)


def generate_random_point_in_country(country_code):
    # Select the polygon for the country
    country_polygon = world[world['iso3'] == country_code].geometry.iloc[0]
    #print(country_polygon)

    minx, miny, maxx, maxy = country_polygon.bounds
    while True:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if country_polygon.contains(random_point):
            return random_point.y, random_point.x  # Return as (latitude, longitude)

start_time = time.time()
print(generate_random_point_in_country('CHE'))  # 'CHE' is the ISO3 country code for Switzerland
end_time = time.time()
execution_time = end_time - start_time
print(f"The function took {execution_time} seconds to execute.")




