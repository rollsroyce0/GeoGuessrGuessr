import geopandas as gpd
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point

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
world['weighted_weighted_counts'].dropna().sort_values(ascending=False).plot(kind='bar', figsize=(15, 10))
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
