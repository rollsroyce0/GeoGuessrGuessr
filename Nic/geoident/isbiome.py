import os
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

class VBRGeoAnalyzer:
    def __init__(self):
        print("Starting VBRGeoAnalyzer...")
        start_time = time.time()
        biomes_shp_file = 'shapefiles\\biomes_large_shp\\wwf_terr_ecos.shp'
        country_json_file = 'shapefiles\\country_json\\world-administrative-boundaries.geojson'
        urban_json_file = 'shapefiles\\urban_shp\\ne_50m_urban_areas.shp'

        # LOAD SHAPEFILES
        script_dir = os.path.dirname(os.path.realpath(__file__))
        biomes_shp_file_path = os.path.join(script_dir, biomes_shp_file)
        country_json_file_path = os.path.join(script_dir, country_json_file)
        urban_shp_file_path = os.path.join(script_dir, urban_json_file)
        self.biomes = gpd.read_file(biomes_shp_file_path)
        self.countries = gpd.read_file(country_json_file_path)
        self.urban = gpd.read_file(urban_shp_file_path)

        # Create spatial indexes
        self.biomes_sindex = self.biomes.sindex
        self.countries_sindex = self.countries.sindex
        self.urban_sindex = self.urban.sindex
        end_time = time.time()
        loading_time = end_time - start_time
        print(f"Shapefiles loaded in {loading_time} seconds.")

    def is_land(self, coords: tuple) -> bool:
        point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

        # Use the spatial index to find the rows that contain the point
        possible_matches_index = list(self.countries_sindex.intersection(point.bounds))
        possible_matches = self.countries.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(point)]

        for _, row in precise_matches.iterrows():
            return True
        return False

    def is_biome(self, coords: tuple) -> str:
        point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

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

        possible_matches_index = list(self.biomes_sindex.intersection(point.bounds))
        possible_matches = self.biomes.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(point)]

        for _, row in precise_matches.iterrows():
            return biome_dict[row['BIOME']]
        return 'Ocean or Desert'

    def is_country(self, coords: tuple) -> dict:
        # Create a point from the coordinates
        point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

        # Use the spatial index to find the rows that contain the point
        possible_matches_index = list(self.countries_sindex.intersection(point.bounds))
        possible_matches = self.countries.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(point)]

        for _, row in precise_matches.iterrows():
            return {'country': row['name'], 'region': row['region'], 'country_code': row['iso3']}
        return {'country': 'No country found', 'region': 'No region found', 'country_code': 'No country code found'}

    def is_urban(self, coords: tuple) -> bool:
        # Create a point from the coordinates
        point = Point(coords[1], coords[0])  # Point takes (longitude, latitude)

        # Use the spatial index to find the rows that contain the point
        possible_matches_index = list(self.urban_sindex.intersection(point.bounds))
        possible_matches = self.urban.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(point)]

        # If any matches are found, return True
        if not precise_matches.empty:
            return True
        return False
    
geo_analyzer = VBRGeoAnalyzer()
print(geo_analyzer.is_land((34.052235, -118.243683)))
print(geo_analyzer.is_biome((34.052235, -118.243683)))
print(geo_analyzer.is_country((34.052235, -118.243683)))
print(geo_analyzer.is_urban((34.052235, -118.243683)))