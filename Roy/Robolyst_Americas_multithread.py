import concurrent.futures
import requests
from PIL import Image
import matplotlib.pyplot as plt
from rich.progress import track
import selenium
from selenium.webdriver.common.by import By
import os
import numpy as np
import time
from global_land_mask import globe
from streetview import search_panoramas
import warnings
from Helper_Functions.geofindcountry import generate_random_country_code, generate_random_point_in_country
from Helper_Functions.geofindurban import generate_random_point_in_urban_area

# Delete all images in the folder to free up space
for image in os.listdir("Roy/images_first_try/"):
    os.remove("Roy/images_first_try/"+image)

start_time = time.time()
warnings.filterwarnings("ignore")
zoom = 3
path_to_folder = "Roy/images_first_try/"

options = selenium.webdriver.ChromeOptions()
options.add_argument('log-level=3')
options.add_argument("--headless")   # Run the browser in the background

driver = selenium.webdriver.Chrome(options=options)
url = "https://www.google.ch/maps/"
driver.get(url)
driver.set_window_size(1920, 1080)

lat_track = []
lon_track = []
counter = 0

# Define a function to process each iteration
def process_location(i):
    global counter
    f_start = time.time()
    country_code_mapping = {
        range(0, 10): 'USA',
        range(10, 20): 'USA',
        range(20, 30): 'AUS',
        range(30, 40): 'RUS',
        range(40, 45): 'NZL',
        range(45, 50): 'CHL',
        range(50, 55): 'RUS',
        range(55, 60): 'AUS',
        range(60, 65): 'AUS',
        range(65, 70): 'USA',
        range(70, 80): 'USA',
        range(80, 85): 'RUS',
        range(85, 100): 'Urban'
    }

    overflow = False
    duds = 0

    while True:
        remainder = i % 100
        code = next((country_code_mapping[key] for key in country_code_mapping if remainder in key), None)
        
        if code is None:
            raise ValueError("Remainder is out of defined range")

        if duds > 20:
            overflow = True
            print("Too many duds")
            break

        if remainder < 5:
            lat, lon = generate_random_point_in_country(code)
        else:
            lat, lon = generate_random_point_in_urban_area()
            code = "Urban"

        if code == 'RUS' and lon > 45 and lat > 59:
            continue
        if code == 'CAN' and lon > 54:
            continue

        panoids = search_panoramas(lat=lat, lon=lon)
        if len(panoids) == 0:
            lat_track.append([lat, 1])
            lon_track.append([lon, 1])
            duds += 1
        else:
            break

    if overflow:
        overflow = False
        print("Time taken for one image:", time.time() - f_start)
        print("Time taken for all images:", time.time() - start_time)
        print("Average time per image:", (time.time() - start_time) / counter)
        return None

    panoid = panoids[0]
    panoid = str(panoid)
    panoid = panoid.split("'")[1]
    panoid = panoid.split("'")[0]

    if len(panoid) != 22:
        print("Invalid panoid")
        return None

    print("Panoid found:", panoid, " for country code:", code)
    counter += 1

    lat_track.append([lat, 0])
    lon_track.append([lon, 0])

    # Download and save the images
    for x in range(2**zoom):
        for y in range(2**(zoom-1)):
            if y == 0 or y == 2**(zoom-1)-1:
                continue
            url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
            response = requests.get(url)
            status_code = response.status_code

            save_path = path_to_folder+str(lat)+"_"+str(lon)+"_Index_"+str(x)+"_"+str(y)+".png"
            if status_code == 200:
                if x == 0 and y == 1:
                    print("Image exists")
            else:
                print("Old Gen", x, y)
                img = Image.new("RGB", (512, 512))
                img.save(save_path)
                continue

            driver.get(url)
            time.sleep(0.1)
            driver.save_screenshot(save_path)
    print("Time taken for one image:", time.time()-f_start)
    print("Time taken for all images:", time.time()-start_time)
    print("Average time per image:", (time.time()-start_time)/counter)

# Use ThreadPoolExecutor to run the process_location function concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    list(track(executor.map(process_location, range(50)), description="Processing locations", total=50))

driver.quit()

# Cropping and combining images
for image in track(os.listdir(path_to_folder), description="Cropping images"):
    img = Image.open(path_to_folder + image)
    width, height = img.size
    left = width / 2 - 256
    top = height / 2 - 256
    right = width / 2 + 256
    bottom = height / 2 + 256

    img = img.crop((left, top, right, bottom))
    img.save(path_to_folder + image)

path_to_combined_folder = "Roy/combined_images/"

# Combine images
for image in track(os.listdir(path_to_folder), description="Combining images"):
    img = image.split("_", 3)
    ind = img[-1]
    ind = ind[0]
    img = img[0] + "_" + img[1] + "_" + img[2] + "_"
    new_image = Image.new("RGB", (1024, 1024))
    x = int(ind)
    for y in [1, 2]:
        image = Image.new("RGB", (512, 512))
        if os.path.exists(path_to_folder + img + str(x) + "_" + str(y) + ".png"):
            image = Image.open(path_to_folder + img + str(x) + "_" + str(y) + ".png")
        new_image.paste(image, (0, (y-1) * 512))

        if x == 2**zoom-1:
            x = -1

        image = Image.new("RGB", (512, 512))
        if os.path.exists(path_to_folder + img + str(x+1) + "_" + str(y) + ".png"):
            image = Image.open(path_to_folder + img + str(x+1) + "_" + str(y) + ".png")
        new_image.paste(image, (512, (y-1) * 512))

        if x == -1:
            x = 2**zoom-1

    x += 1
    if x > 2**zoom-1:
        x = x - 2**zoom

    new_image.save(path_to_combined_folder + img + "_Index_" + str(x+1) + ".png")

lat_track = np.array(lat_track)
lon_track = np.array(lon_track)

plt.scatter(lon_track[lon_track[:,1]==2][:,0], lat_track[lon_track[:,1]==2][:,0], c="blue", label="Excluded for No coverage", s=20)
plt.scatter(lon_track[lon_track[:,1]==1][:,0], lat_track[lon_track[:,1]==1][:,0], c="red", label="No Panoids", s=20)
plt.scatter(lon_track[lon_track[:,1]==0][:,0], lat_track[lon_track[:,1]==0][:,0], c="green", label="Found a spot", s=100)
plt.legend()
plt.show()

print("Done")
