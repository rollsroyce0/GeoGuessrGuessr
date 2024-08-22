import geopandas as gpd
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import numpy as np
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Function to extract coordinates from the image path
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

# Extract coordinates and create a GeoDataFrame
data = []
for i in os.listdir("D:/GeoGuessrGuessr/geoguesst"):
    lat, lon = extract_coordinates(i)
    data.append([lat, lon])

# Convert the coordinates into a GeoDataFrame
gdf = gpd.GeoDataFrame(data, columns=['Latitude', 'Longitude'])
gdf['geometry'] = [Point(lon, lat) for lat, lon in data]

# Load the world shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Perform a spatial join to count the number of points per country
gdf = gpd.sjoin(gdf, world, how="left", op="within")
country_counts = gdf['name'].value_counts()

# Merge country counts with the world data for proper matching
world = world.set_index('name').join(country_counts.rename('counts'))

# Weigh each country's count by the country's area (area is in square degrees)
world['area'] = world['geometry'].to_crs({'proj':'cea'}).area  # Convert to an equal-area projection
world['weighted_counts'] = world['counts'] / world['area']
world["weighted_weighted_counts"] = world["weighted_counts"] * world["counts"]**2

# Display the per-country weighted count
print(world['counts'])

# Plot the weighted country counts
world['weighted_counts'].dropna().sort_values(ascending=False).plot(kind='bar', figsize=(15, 10))
plt.title('Weighted Country Counts by Area')
plt.yscale('log')
plt.ylabel('Weighted Count')
plt.xlabel('Country')
plt.show()

# Plot the datapoints on the world map
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgray')
gdf.plot(ax=ax, color='red', markersize=10, label="Data Points")

# Set the title and show the plot
ax.set_title('Datapoints on World Map')
plt.legend()
plt.show()


# redefine the GeoDataFrame to include only the coordinates of the data in Roy/combined_images
new_data = []
for i in os.listdir("Roy/combined_images"):
    lat, lon = extract_coordinates(i)
    new_data.append([lat, lon])
    

#gdf = gpd.GeoDataFrame(new_data, columns=['Latitude', 'Longitude'])

longitudes = gdf['Longitude']
latitudes = gdf['Latitude']

longitudes = np.array(longitudes) [::100]
latitudes = np.array(latitudes)[::100]
print(len(longitudes))

# Create a Gaussian KDE for the heatmap
xy = np.vstack([longitudes, latitudes])
kde = gaussian_kde(xy, bw_method='silverman')

# Create a grid to evaluate the KDE
lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 1000), np.linspace(-90, 90, 500))
positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
density = np.reshape(kde(positions).T, lon_grid.shape)

# Plot
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

# Add features to the map
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Plot the heatmap
plt.contourf(lon_grid, lat_grid, density, cmap='hot', levels=np.linspace(0, density.max(), 100), transform=ccrs.PlateCarree())
plt.colorbar(label='Density')

plt.title('Heatmap of Coordinates')
plt.show()
