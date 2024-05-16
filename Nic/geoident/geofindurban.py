import random
import numpy as np
from shapely.geometry import Polygon, Point
import geopandas as gpd
import os
from shapely.ops import unary_union
from shapely.affinity import translate
import random

# Load the shapefile
country_json_file = 'shapefiles\\urban_shp\\ne_50m_urban_areas.shp'
script_dir = os.path.dirname(os.path.realpath(__file__))
shp_file_path = os.path.join(script_dir, country_json_file)
urban_areas = gpd.read_file(shp_file_path)

def select_random_polygon(gdf):
    random_index = random.randint(0, len(gdf) - 1)
    return gdf.iloc[random_index].geometry

def generate_random_point_in_polygon(poly):
    # Find the "pole of inaccessibility" (most distant internal point from the polygon's edges)
    pole_of_inaccessibility = poly.representative_point()

    # Generate a random point within the polygon
    while True:
        # Generate a random distance and angle
        r = random.random()
        theta = 2 * np.pi * random.random()
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)

        # Translate the pole of inaccessibility by the random distance and angle
        random_point = translate(pole_of_inaccessibility, dx, dy)

        # If the random point is still within the polygon, return it
        if poly.contains(random_point):
            return random_point

def generate_random_point_in_urban_area():
    urban_polygon = select_random_polygon(urban_areas)
    random_point = generate_random_point_in_polygon(urban_polygon)
    return random_point.y, random_point.x