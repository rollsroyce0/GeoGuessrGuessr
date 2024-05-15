import requests
from PIL import Image
import matplotlib.pyplot as plt
from rich.progress import track
import selenium
from selenium.webdriver.common.by import By
import os
import numpy as np
import time
from selenium.webdriver.common.action_chains import ActionChains
from global_land_mask import globe
import warnings

# delete all images in the folder
for image in os.listdir("Roy/images_first_try/"):
    os.remove("Roy/images_first_try/"+image)


warnings.filterwarnings("ignore")

lat_track=[]
lon_track = []

for i in track(range(300000)):
    # generate random latitude and longitude within street view limits
    lat = np.random.uniform(-70,80)
    lon = np.random.uniform(-180,180)
    
    # rule out China
    if lat >29 and lat <42 and lon > 85 and lon < 120:
        #print("China")
        lat_track.append([lat, 0])
        lon_track.append([lon, 0])
        continue

    
    # check if the coordinates are on land
    if not (globe.is_land(lat, lon)):
        #print("Not on land")
        lat_track.append([lat, 0])
        lon_track.append([lon, 0])
        continue
    #print(lat, lon)
    
    lat_track.append([lat, 1])
    lon_track.append([lon, 1])




lat_track = np.array(lat_track)
lon_track = np.array(lon_track)

plt.scatter(lon_track[lon_track[:,1]==1][:,0], lat_track[lon_track[:,1]==1][:,0], c="blue", label="On land", s=1)
plt.scatter(lon_track[lon_track[:,1]==0][:,0], lat_track[lon_track[:,1]==0][:,0], c="red", label="Not on land", s=1)

plt.show()
print("Done")
    
