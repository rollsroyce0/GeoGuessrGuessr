"""
Consolidated Geographic Utility Functions

This module combines geographic finding functions from both Roy/Helper_Functions and Nic/geoident directories,
eliminating redundancy while preserving all functionality.
"""

import os
import time
import geopandas as gpd
from shapely.geometry import Point
import random
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import translate

# Load geographic data files
script_dir = os.path.dirname(os.path.realpath(__file__))

def _load_country_data():
    """Load country boundary data with spatial index"""
    country_json_file = 'shapefiles\\country_json\\world-administrative-boundaries.geojson'
    shp_file_path = os.path.join(script_dir, country_json_file)
    world = gpd.read_file(shp_file_path)
    return world, world.sindex

def _load_urban_data():
    """Load urban area data with spatial index"""
    urban_json_file = 'shapefiles\\urban_shp\\ne_50m_urban_areas.shp'
    shp_file_path = os.path.join(script_dir, urban_json_file)
    urban_areas = gpd.read_file(shp_file_path)
    return urban_areas, urban_areas.sindex

# Country-related functions
def generate_random_country_code(continent=None):
    """
    Generate a random country code, optionally filtered by continent

    Args:
        continent (str, optional): Continent name to filter by. If None, selects from all countries.

    Returns:
        str: Random ISO3 country code
    """
    world, _ = _load_country_data()

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
    """
    Generate a random point within a specific country's boundaries

    Args:
        country_code (str): ISO3 country code

    Returns:
        tuple: (latitude, longitude) coordinates
    """
    world, _ = _load_country_data()

    # Select the polygon for the country
    country_polygon = world[world['iso3'] == country_code].geometry.iloc[0]

    minx, miny, maxx, maxy = country_polygon.bounds
    while True:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if country_polygon.contains(random_point):
            return random_point.y, random_point.x  # Return as (latitude, longitude)

# Urban area functions
def select_random_polygon(gdf):
    """
    Select a random polygon from a GeoDataFrame

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing polygons

    Returns:
        shapely.geometry.polygon.Polygon: Randomly selected polygon
    """
    random_index = random.randint(0, len(gdf) - 1)
    return gdf.iloc[random_index].geometry

def generate_random_point_in_polygon(poly):
    """
    Generate a random point within a polygon

    Args:
        poly (shapely.geometry.polygon.Polygon): Input polygon

    Returns:
        shapely.geometry.point.Point: Random point within the polygon
    """
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

def generate_random_point_in_urban_area(n=None):
    """
    Generate random point(s) in urban areas

    Args:
        n (int, optional): Number of points to generate. If None, generates one point.

    Returns:
        tuple or list: Single (lat, lon) tuple if n=None, else list of tuples
    """
    urban_areas, _ = _load_urban_data()

    if n is None:
        urban_polygon = select_random_polygon(urban_areas)
        random_point = generate_random_point_in_polygon(urban_polygon)
        return random_point.y, random_point.x
    else:
        coordinates = []
        for _ in range(n):
            urban_polygon = select_random_polygon(urban_areas)
            random_point = generate_random_point_in_polygon(urban_polygon)
            coordinates.append((random_point.y, random_point.x))
        return coordinates

# Test functions (can be removed in production)
if __name__ == "__main__":
    print("Testing geographic_utils.py...")

    # Test country functions
    print("Random country code:", generate_random_country_code())
    print("Random point in Switzerland:", generate_random_point_in_country('CHE'))

    # Test urban functions
    print("Random urban point:", generate_random_point_in_urban_area())
    print("5 random urban points:", generate_random_point_in_urban_area(5))