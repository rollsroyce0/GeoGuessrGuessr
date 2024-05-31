import os
import geopandas as gpd
from shapely.geometry import Point

biomes_shp_file = 'shapefiles\\biomes_large_shp\\wwf_terr_ecos.shp'

# Load the shapefile and create a spatial index
script_dir = os.path.dirname(os.path.realpath(__file__))
shp_file_path = os.path.join(script_dir, biomes_shp_file)
world = gpd.read_file(shp_file_path)
sindex = world.sindex

biome_dict = {
    1: 'Tropical & Subtropical Moist Broadleaf Forests',
    2: 'Tropical & Subtropical Dry Broadleaf Forests',
    3: 'Tropical & Subtropical Coniferous Forests',
    4: 'Temperate Broadleaf & Mixed Forests',
    5: 'Temperate Conifer Forests',
    6: 'Boreal Forests/Taiga',
    7: 'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    8: 'Temperate Grasslands, Savannas & Shrublands',
    9: 'Flooded Grasslands & Savannas',
    10: 'Montane Grasslands & Shrublands',
    11: 'Tundra',
    12: 'Mediterranean Forests, Woodlands & Scrub',
    13: 'Deserts & Xeric Shrublands',
    14: 'Mangroves'
}

def is_biome(coords):
    point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

    # Use the spatial index to find the rows that contain the point
    possible_matches_index = list(sindex.intersection(point.bounds))
    possible_matches = world.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(point)]

    for _, row in precise_matches.iterrows():
        return biome_dict[row['BIOME']]

    return 'Ocean or Desert'

coords = (47.3779506,8.534353)
print(is_biome(coords))