"""
Consolidated Road Utility Functions

This module combines road finding functions from both Roy/Helper_Functions and Nic/geoclosest directories,
eliminating redundancy while preserving all functionality.
"""

import requests
import json
from geopy.distance import geodesic

def find_closest_road(lat, lon):
    """
    Find the closest road to given coordinates using Overpass API

    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate

    Returns:
        tuple: ((road_lat, road_lon), distance_in_meters)
    """
    # Overpass API url
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Overpass QL query - search for highways within 1000 meters
    overpass_query = f"""
    [out:json];
    way(around:1000,{lat},{lon})["highway"];
    (._;>;);
    out body;
    """

    # Set required headers for Overpass API
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'text/plain',
        'User-Agent': 'GeoGuessrGuessr/1.0',
        'Referer': 'https://github.com/GeoGuessrGuessr'
    }

    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()

        # Extract coordinates of the nearest road
        if data.get('elements', []):
            for element in data['elements']:
                if element['type'] == 'way':
                    # Get the first node of the way as the location of the road
                    road_lat = data['elements'][0]['lat']
                    road_lon = data['elements'][0]['lon']
                    break
            else:
                # If no road is found, return the input coordinates
                road_lat, road_lon = lat, lon
        else:
            # If no road is found, return the input coordinates
            road_lat, road_lon = lat, lon

        distance = geodesic((lat, lon), (road_lat, road_lon)).meters

        return (road_lat, road_lon), distance

    except requests.exceptions.RequestException as e:
        print(f"Error calling Overpass API: {e}")
        return (lat, lon), 0.0
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing API response: {e}")
        return (lat, lon), 0.0

# Test function (can be removed in production)
if __name__ == "__main__":
    print("Testing road_utils.py...")

    # Test with known locations
    test_locations = [
        (51.288, 13.91),  # Brandenburg Gate, Berlin
        (40.7128, -74.0060),  # New York City
        (0, 0)  # Gulf of Guinea (should return original coords)
    ]

    for lat, lon in test_locations:
        road_coords, distance = find_closest_road(lat, lon)
        print(f"Location ({lat}, {lon}): closest road at {road_coords}, distance: {distance:.1f}m")