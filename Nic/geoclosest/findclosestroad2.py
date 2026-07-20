from Roy.Helper_Functions.road_utils import find_closest_road


if __name__ == "__main__":
    lat, lon = 51.288, 13.91
    road_coords, distance = find_closest_road(lat, lon)
    print(f"The closest road is at {road_coords} and is {distance} meters away.")