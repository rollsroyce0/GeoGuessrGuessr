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
    print(data)

    # Extract coordinates of the nearest road
    if data['elements'] != []:
        for element in data['elements']:
            if element['type'] == 'way':
                # Get the first node of the way as the location of the road
                print(data['elements'][0])
                road_lat = data['elements'][0]['lat']
                road_lon = data['elements'][0]['lon']
                
                break
        distance = geodesic((lat, lon), (road_lat, road_lon)).meters
    else:
        # If no road is found, return the input coordinates
        road_lat, road_lon = lat, lon
        distance = 1000000 # A large distance  

    return (road_lat, road_lon), distance

lat, lon = 0,0  # Coordinates of Brandenburg Gate, Berlin
road_coords, distance = find_closest_road(lat, lon)
print(f"The closest road is at {road_coords} and is {distance} meters away.")