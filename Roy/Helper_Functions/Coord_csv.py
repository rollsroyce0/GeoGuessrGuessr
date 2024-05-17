import os
import numpy as np
import pandas as pd
# load the names of the images in other folder different drive

images = os.listdir("D:/GeoGuessrGuessr/geoguesst")

lat =[image.split("_")[0] for image in images]
lon = [image.split("_")[1] for image in images]

lat = np.array(lat, dtype=float)
lon = np.array(lon, dtype=float)
print(len(lat)/8)

# Create a dataframe
df = pd.DataFrame({"Latitude": lat, "Longitude": lon})

save_path = "Roy/combined_images.csv"
df.to_csv(save_path, index=False)

