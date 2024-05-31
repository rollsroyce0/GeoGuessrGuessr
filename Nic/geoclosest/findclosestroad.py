import requests
import json
from geopy.distance import geodesic

def find_closest_road(lat, lon):
    # Overpass API url
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Overpass QL query
    overpass_query = f"""
    [out:json];
    way(around:1000,{lat},{lon})["highway"];
    (._;>;);
    out body;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    # Extract coordinates of the nearest road
    for element in data['elements']:
        if element['type'] == 'way':
            # Get the first node of the way as the location of the road
            road_lat = data['elements'][0]['lat']
            road_lon = data['elements'][0]['lon']
            break

    # Calculate the distance
    distance = geodesic((lat, lon), (road_lat, road_lon)).meters

    return (road_lat, road_lon), distance

lat, lon = 51.288, 13.91  # Coordinates of Brandenburg Gate, Berlin
road_coords, distance = find_closest_road(lat, lon)
print(f"The closest road is at {road_coords} and is {distance} meters away.")