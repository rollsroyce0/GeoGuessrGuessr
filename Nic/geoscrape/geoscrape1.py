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
from streetview import search_panoramas
import warnings

script_dir = os.path.dirname(os.path.realpath(__file__))
images_dir = os.path.join(script_dir, 'temp_images')
# delete all images in the folder
for image in os.listdir(images_dir):
    os.remove(images_dir+image)

warnings.filterwarnings("ignore")
zoom = 3

options = selenium.webdriver.ChromeOptions()
options.add_argument('log-level=3')
options.add_argument("--headless")

driver = selenium.webdriver.Chrome(options=options)
url = "https://www.google.ch/maps/"
driver.get(url)
driver.set_window_size(1920, 1080)
buttons = driver.find_elements(By.CSS_SELECTOR, "button")
#print(buttons)
buttons[1].click()

lat_track=[]
lon_track = []

for i in track(range(7000)):
    # generate random latitude and longitude within street view limits
    lat = np.random.uniform(-65,80)
    lon = np.random.uniform(-180,180)
    
    # rule out China
    if lat >29 and lat <42 and lon > 85 and lon < 120:
        #print("China")
        lat_track.append([lat, 2])
        lon_track.append([lon, 2])
        continue
    
    #rule out Greenland
    if lat > 67 and lat < 80 and lon > -49 and lon < -27:
        lat_track.append([lat, 2])
        lon_track.append([lon, 2])
        continue
        
    # check if the coordinates are on land
    if not (globe.is_land(lat, lon)):
        #print("Not on land")
        lat_track.append([lat, 2])
        lon_track.append([lon, 2])
        continue
    #print(lat, lon)
    
    panoids = search_panoramas(lat = lat, lon = lon)
    if len(panoids) == 0:
        #print("No panoids found")
        lat_track.append([lat, 1])
        lon_track.append([lon, 1])
        continue
    #print(panoids)
    
    panoid = panoids[0]
    panoid = str(panoid)
    panoid = panoid.split("'")[1]
    panoid = panoid.split("'")[0]
    
    print("Panoid found:", panoid)
    
    lat_track.append([lat, 0])
    lon_track.append([lon, 0])

    # get the images
    
    for x in range(2**zoom):
        for y in range(2**(zoom-1)):
            if y == 0 or y == 2**(zoom-1)-1:
                #print(y)
                continue
            url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
            
            #check if the image exists
            response = requests.get(url)
            status_code = response.status_code
            
            save_path = images_dir+str(lat)+"_"+str(lon)+"_Index_"+str(x)+"_"+str(y)+".png"
            if status_code == 200:
                #print(panoid)
                if x == 0 and y == 1:
                    print("Image exists")
            else:
                print("Old Gen", x, y)
                # save a blank image
                img = Image.new("RGB", (512, 512))
                img.save(save_path)
                continue
                
            
            driver.get(url)
            # delay to load the page
            time.sleep(0.25)
            
            driver.save_screenshot(save_path)
        
driver.quit()

print("zoom", zoom)
# Since the images have a massive black border around them, we need to crop them
# to the actual street view image
for image in os.listdir(images_dir):
    img = Image.open(images_dir+image)
    width, height = img.size
    left = width/2 -256
    top = height/2 -256
    right = width/2 +256
    bottom = height/2 +256
    
    
    img = img.crop((left, top, right, bottom))
    img.save(images_dir+image)


path_to_combined_folder = "Roy/combined_images/"
# combine 4 images into 1
for image in track(os.listdir(images_dir)):
    img = image.split("_",3)
    ind = img[-1]
    ind = ind[0]
    img = img[0]+"_"+img[1]+"_"+img[2]+"_"
    #print(img)
    #print(ind)
    new_image = Image.new("RGB", (1024, 1024))
    x = int(ind)
    for y in [1,2]:
            #check if the image exists
            #default image as a black image
            image = Image.new("RGB", (512, 512))
            if os.path.exists(images_dir+img+str(x)+"_"+str(y)+".png"):
                image = Image.open(images_dir+img+str(x)+"_"+str(y)+".png")
            new_image.paste(image, (0, (y-1)*512))
            
            # handle wraparound
            if x == 2**zoom-1:
                x=-1
            
                
            image = Image.new("RGB", (512, 512))
            if os.path.exists(images_dir+img+str(x+1)+"_"+str(y)+".png"):
                image = Image.open(images_dir+img+str(x+1)+"_"+str(y)+".png")
            new_image.paste(image, (512, (y-1)*512))
            
            if x == -1:
                x = 2**zoom-1
    
    x+=1
    if x > 2**zoom-1:
        x = x - 2**zoom
    
    image = img
    new_image.save(path_to_combined_folder+image+"_Index_"+str(x+1)+".png")


lat_track = np.array(lat_track)
lon_track = np.array(lon_track)

plt.scatter(lon_track[lon_track[:,1]==2][:,0], lat_track[lon_track[:,1]==2][:,0], c="blue", label="China or Ocean", s=40)
plt.scatter(lon_track[lon_track[:,1]==1][:,0], lat_track[lon_track[:,1]==1][:,0], c="red", label="No Panoids", s=40)
plt.scatter(lon_track[lon_track[:,1]==0][:,0], lat_track[lon_track[:,1]==0][:,0], c="green", label="Found a spot", s=150)
plt.legend()

plt.show()
print("Done")
    
