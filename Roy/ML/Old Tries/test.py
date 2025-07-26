import numpy as np
import pandas as pd

def assign_continent(lat, lon):
    if lat < 30 and -30 <= lon <= 60:
        return 0  # Africa
    elif lat > -13 and lon > 45:
        return 1  # Asia
    elif -50 < lat < -13 and 110 <= lon <= 180:
        return 2  # Australia
    elif lat > 12 and -130 <= lon <= -30:
        return 3  # North America
    elif lat < 12 and -90 <= lon <= -30:
        return 4  # South America
    elif lat > 30 and -30 <= lon <= 45:
        return 5  # Europe
    else:
        return 6  # Others (e.g., Pacific islands, Antarctica)
    
# Randomly generate some coordinates
np.random.seed(0)
n_points = 50000
latitudes = np.random.uniform(-90, 90, n_points)
longitudes = np.random.uniform(-180, 180, n_points)
coordinates = list(zip(latitudes, longitudes))

# Assign continent labels to coordinates
continent_labels = np.array([assign_continent(lat, lon) for lat, lon in coordinates])
print(continent_labels)

# plot the coordinates on a world map, color-coded by continent

# Create a DataFrame with the coordinates and continent labels
df = pd.DataFrame(coordinates, columns=['Latitude', 'Longitude'])
df['Continent'] = continent_labels

# Map the continent labels to continent names
continent_names = ['Africa', 'Asia', 'Australia', 'North America', 'South America', 'Europe', 'Others']
df['Continent'] = df['Continent'].map({i: name for i, name in enumerate(continent_names)})
print(df)

# Plot the coordinates on a world map, color-coded by continent
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a GeoDataFrame from the DataFrame with coordinates
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))

# Plot the world map
fig, ax = plt.subplots(figsize=(12, 8))
world.boundary.plot(ax=ax, linewidth=1)
gdf.plot(ax=ax, markersize=5, column='Continent', legend=True, legend_kwds={'loc': 'upper left'})
plt.title('Coordinates Mapped to Continents')
plt.show()