import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

def is_in_ocean(coord: tuple) -> bool:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    shp_file_path = os.path.join(script_dir, 'shapefiles\\land_shp\\ne_110m_land.shp')
    world = gpd.read_file(shp_file_path)
    point = Point(coord)
    for _, row in world.iterrows():
        if row['geometry'].contains(point):
            #print('The coordinates are not in the ocean.')
            return False
    #print('The coordinates are in the ocean.')
    return True

def plot_is_in_ocean():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    shp_file_path = os.path.join(script_dir, 'ne_110m_land.shp')
    world = gpd.read_file(shp_file_path)
    world.plot()
    plt.show()
