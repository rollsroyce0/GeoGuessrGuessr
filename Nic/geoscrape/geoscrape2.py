import requests
import sys
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
from streetview import search_panoramas
import warnings
sys.path.append('../')
import geoident.geofindurban

script_dir = os.path.dirname(os.path.realpath(__file__))
images_dir = os.path.join(script_dir, 'temp_images')
for image in os.listdir(images_dir):
    os.remove(images_dir+'\\'+image)

warnings.filterwarnings("ignore")
zoom = 3
options = selenium.webdriver.ChromeOptions()
options.add_argument('log-level=3')
options.add_argument("--headless")

def generate_random_coords():
    lat, lon = geoident.geofindurban.generate_random_point_in_urban_area()
    return lat, lon

def find_panoid(lat, lon):
    try:
        panoids = search_panoramas(lat = lat, lon = lon)
        if len(panoids) != 0:
            panoid = str(panoids[0]).split("'")[1].split("'")[0]  # Corrected here
            # print(f"Panoid found: {panoid}")
            return panoid
            # print("No panoid found")
    except Exception as e:
        print(f"No panoid found: {e}")
    return None

def get_images(driver, lat, lon, panoid, zoom):
    print(".", end = "")
    for x in range(2**zoom):
        y = 1
        url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
        response = requests.get(url)
        status_code = response.status_code
        save_path = images_dir+'\\'+str(lat)+"_"+str(lon)+"_Index_"+str(x)+"_"+str(y)+".png"
        if status_code == 200 and x == 0 and y == 1:
            pass
            #print("New image saved...")
        else:
            #print("Old image saved...")
            img = Image.new("RGB", (512, 512))
            img.save(save_path)
        driver.get(url)
        time.sleep(0.5)
        driver.save_screenshot(save_path)
        # print("Cropping image...")
        img = Image.open(save_path)
        width, height = img.size
        left = width/2 -256
        top = height/2 -256
        right = width/2 +256
        bottom = height/2 +256

        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(save_path)
        # print("Cropped image saved...")

def plot_locations(lat_track, lon_track):
    lat_track = np.array(lat_track)
    lon_track = np.array(lon_track)

    plt.scatter(lon_track[lon_track[:,1]==2][:,0], lat_track[lon_track[:,1]==2][:,0], c="blue", label="China or Ocean", s=40)
    plt.scatter(lon_track[lon_track[:,1]==1][:,0], lat_track[lon_track[:,1]==1][:,0], c="red", label="No Panoids", s=40)
    plt.scatter(lon_track[lon_track[:,1]==0][:,0], lat_track[lon_track[:,1]==0][:,0], c="green", label="Found a spot", s=150)
    plt.legend()

    plt.show()
    print("Done")

def main():
    driver = selenium.webdriver.Chrome(options=options)
    url = "https://www.google.ch/maps/"
    driver.get(url)
    driver.set_window_size(1920, 1080)
    buttons = driver.find_elements(By.CSS_SELECTOR, "button")
    buttons[1].click()

    lat_track=[]
    lon_track = []
    for i in track(range(1000), description="Processing..."):
        panoid = None
        i = 0
        while panoid is None:
            lat, lon = generate_random_coords()
            panoid = find_panoid(lat, lon)
            i += 1
        print(i)
        lat_track.append([lat])
        lon_track.append([lon])
        get_images(driver, lat, lon, panoid, zoom)
    driver.quit()
    plot_locations(lat_track, lon_track)

if __name__ == "__main__":
    main()