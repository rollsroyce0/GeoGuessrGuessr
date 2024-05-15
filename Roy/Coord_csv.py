import os
import numpy as np
import pandas as pd
# load the names of the images in combined_images
images = os.listdir("Roy/combined_images/")

lat =[image.split("_")[0] for image in images]
lon = [image.split("_")[1] for image in images]

lat = np.array(lat, dtype=float)
lon = np.array(lon, dtype=float)
print(len(lat))

# Create a dataframe
df = pd.DataFrame({"Latitude": lat, "Longitude": lon})

save_path = "Roy/combined_images.csv"
df.to_csv(save_path, index=False)

