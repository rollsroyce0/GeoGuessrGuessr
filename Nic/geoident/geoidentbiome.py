import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

import geoidentocean

biomes_shp_file = 'wwf_terr_ecos.shp'

def is_biome(coords):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    shp_file_path = os.path.join(script_dir, biomes_shp_file)
    world = gpd.read_file(shp_file_path)
    point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)
    # plot_biome(coords)

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

    for _, row in world.iterrows():
        if row['geometry'].contains(point):
            return biome_dict[row['BIOME']]
    if geoidentocean.is_in_ocean(coords):
        return 'Ocean'
    else:
        return 'Desert'

def plot_biome(coords):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    shp_file_path = os.path.join(script_dir, biomes_shp_file)
    world = gpd.read_file(shp_file_path)
    world.plot(column='BIOME', legend=True)
    plt.plot(coords[1], coords[0], 'ro')
    plt.show()

coords = (47.3779506,8.534353)
print(is_biome(coords))