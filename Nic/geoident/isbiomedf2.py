import os
import time
import geopandas as gpd
import pandas as pd
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

    def pd_biome(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'longitude' not in df.columns or 'latitude' not in df.columns:
            raise ValueError("DataFrame must contain 'longitude' and 'latitude' columns.")
        print("Columns in the DataFrame: ", df.columns.tolist())

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

        # Spatial Joins
        df['is_land'] = gpd.sjoin(gdf, self.countries, how='left', predicate='within').index.notnull()
        df['is_biome'] = gpd.sjoin(gdf, self.biomes, how='left', predicate='within')['BIOME'].map(self.biome_dict)
        df['is_country'] = gpd.sjoin(gdf, self.countries, how='left', predicate='within')['name']
        df['is_urban'] = gpd.sjoin(gdf, self.urban, how='left', predicate='within').index.notnull()
        return df
    
    def pd_plot(self, df: pd.DataFrame = None):
        fig, ax = plt.subplots()
        self.biomes['BIOME'] = self.biomes['BIOME'].map(self.biome_dict)
        self.biomes.plot(column='BIOME', ax=ax, legend=True, cmap='tab20')
        
        # Plot the urban areas on top
        self.urban.plot(ax=ax, color='black', label='Urban Areas')
        
        # Create a list of patches for the legend
        from matplotlib.patches import Patch
        legend_labels = sorted(self.biome_dict.values()) + ['Urban Areas']
        legend_patches = [Patch(color=c, label=l) for c, l in zip(list(plt.cm.tab20.colors[:len(self.biome_dict)]) + ['black'], legend_labels)]
        
        # Create the legend
        ax.legend(handles=legend_patches, prop={'size': 6})
        
        if 'longitude' in df.columns and 'latitude' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
            gdf.plot(ax=ax, color='red', label='Coordinates', markersize=0.3)

        plt.show()

script_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = 'coords.csv'
df = pd.read_csv(os.path.join(script_dir, csv_path), names=['latitude', 'longitude'])
df = df.drop_duplicates()
start_time = time.time()

geo_analyzer = VBRGeoAnalyzer()
geo_analyzer.pd_biome(df)

end_time = time.time()
print(df.head())
print(len(df), " coordinate entries processed in ", end_time - start_time, " seconds.")
df.to_csv(os.path.join(script_dir, 'output.csv'), index=False)

geo_analyzer.pd_plot(df)