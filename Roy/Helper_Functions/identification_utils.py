"""
Consolidated Geographic Identification Utility Functions

This module combines geographic identification functions from both Roy/Helper_Functions and Nic/geoident directories,
eliminating redundancy while preserving all functionality.
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Load geographic data files
script_dir = os.path.dirname(os.path.realpath(__file__))

def _load_urban_identification_data():
    """Load urban area data for identification purposes"""
    country_json_file = 'shapefiles\\urban_shp\\ne_50m_urban_areas.shp'
    shp_file_path = os.path.join(script_dir, country_json_file)
    urban_areas = gpd.read_file(shp_file_path)
    return urban_areas, urban_areas.sindex

def _load_land_data():
    """Load land data for ocean identification"""
    shp_file_path = os.path.join(script_dir, 'shapefiles\\land_shp\\ne_110m_land.shp')
    world = gpd.read_file(shp_file_path)
    return world

# Urban identification functions
def is_urban(coords):
    """
    Check if coordinates are within an urban area

    Args:
        coords (tuple): (latitude, longitude) coordinates

    Returns:
        bool: True if coordinates are in an urban area, False otherwise
    """
    urban_areas, sindex = _load_urban_identification_data()

    # Create a point from the coordinates
    point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

    # Use the spatial index to find the rows that contain the point
    possible_matches_index = list(sindex.intersection(point.bounds))
    possible_matches = urban_areas.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(point)]

    # If any matches are found, return True
    return not precise_matches.empty

# Ocean/land identification functions
def is_in_ocean(coord):
    """
    Check if coordinates are in the ocean (not on land)

    Args:
        coord (tuple): (latitude, longitude) coordinates

    Returns:
        bool: True if coordinates are in the ocean, False if on land
    """
    world = _load_land_data()
    point = Point(coord)

    for _, row in world.iterrows():
        if row['geometry'].contains(point):
            return False  # On land

    return True  # In ocean

def plot_land_mass():
    """
    Plot land masses for visualization (development/debugging purpose)
    """
    world = _load_land_data()
    world.plot()
    plt.show()

# Test coordinates for urban identification
TEST_COORDINATES = [
    (47.3779506, 8.534353),  # Zurich, Switzerland
    (34.052235, -118.243683),  # Los Angeles, USA
    (-33.865143, 151.209900),  # Sydney, Australia
    (55.755826, 37.617600),  # Moscow, Russia
    (-22.906847, -43.172897),  # Rio de Janeiro, Brazil
    (35.689487, 139.691711),  # Tokyo, Japan
    (28.613939, 77.209023),  # New Delhi, India
    (-1.286389, 36.817223),  # Nairobi, Kenya
    (64.126520, -21.817439),  # Reykjavik, Iceland
    (-25.263740, -57.575926),  # Asuncion, Paraguay
    (32.9681685240936, -120.75474968007332)  # Ocean
]

# Test functions (can be removed in production)
if __name__ == "__main__":
    print("Testing identification_utils.py...")

    # Test urban identification
    print("Urban identification tests:")
    for i, coord in enumerate(TEST_COORDINATES[:5]):  # Test first 5 coordinates
        print(f"  {coord}: {'Urban' if is_urban(coord) else 'Not Urban'}")

    # Test ocean identification
    print("\nOcean identification tests:")
    test_coords = [
        (32.9681685240936, -120.75474968007332),  # Ocean
        (40.7128, -74.0060),  # New York City (land)
        (0, 0)  # Gulf of Guinea (ocean)
    ]
    for coord in test_coords:
        print(f"  {coord}: {'Ocean' if is_in_ocean(coord) else 'Land'}")