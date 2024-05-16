import random
import time
import numpy as np
from shapely.geometry import Polygon, Point
from typing import Optional
import geopandas as gpd
import os
from shapely.ops import unary_union
from shapely.affinity import translate
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm

class VBRGeoFinder:
    def __init__(self):
        print("Starting VBRGeoFinder...")
        start_time = time.time()
        biomes_shp_file = 'shapefiles\\biomes_large_shp\\wwf_terr_ecos.shp'
        country_json_file = 'shapefiles\\country_json\\world-administrative-boundaries.geojson'
        urban_json_file = 'shapefiles\\urban_shp\\ne_50m_urban_areas.shp'

        # LOAD SHAPEFILES
        script_dir = os.path.dirname(os.path.realpath(__file__))
        biomes_shp_file_path = os.path.join(script_dir, biomes_shp_file)
        country_json_file_path = os.path.join(script_dir, country_json_file)
        urban_shp_file_path = os.path.join(script_dir, urban_json_file)
        self.biomes = gpd.read_file(biomes_shp_file_path).set_crs("EPSG:4326")
        self.countries = gpd.read_file(country_json_file_path).set_crs("EPSG:4326")
        self.urban = gpd.read_file(urban_shp_file_path).set_crs("EPSG:4326")
        self.biome_dict = {
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

        end_time = time.time()
        loading_time = end_time - start_time
        print(f"Shapefiles loaded in {loading_time} seconds.")

    def generate_point(self, urban: Optional[bool] = False, country: Optional[str] = None, biome: Optional[int] = None, n: Optional[int] = 1, plot: Optional[bool] = False):
        max_attempts = 1000
        points = []
        for _ in tqdm(range(n), desc="Generating points"):
            polygons = []
            if urban:
                polygons.append(self.urban.unary_union)
            if country:
                polygons.append(self.countries[self.countries['iso3'] == country].unary_union)
            if biome:
                polygons.append(self.biomes[self.biomes['BIOME'] == biome].unary_union)
            if polygons:
                intersection_area = polygons[0]
                for poly in polygons[1:]:
                    intersection_area = intersection_area.intersection(poly)
            else:
                intersection_area = self.countries.unary_union
            for _ in range(max_attempts):
                point = Point(random.uniform(intersection_area.bounds[0], intersection_area.bounds[2]), random.uniform(intersection_area.bounds[1], intersection_area.bounds[3]))
                if intersection_area.contains(point):
                    points.append(point)
                    break
            else:
                print("Did not find a point that satisfies the requirements")
                return None

        # Plotting
        if plot:
            fig, ax = plt.subplots()
            gpd.GeoSeries(intersection_area).plot(ax=ax, color='red')
            gpd.GeoSeries(points).plot(ax=ax, color='blue', markersize=5)
            plt.show()

        return points
    
finder = VBRGeoFinder()
print(finder.generate_point(urban=True, country='CHE', n=20, plot = True))