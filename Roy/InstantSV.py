import requests
from PIL import Image
from rich.progress import track
import selenium
from selenium.webdriver.common.by import By
import os
import numpy as np
import time
from selenium.webdriver.common.action_chains import ActionChains
from global_land_mask import globe
import warnings

# Maybe always delete first try to free up space and reduce computation time

# delete all images in the folder
for image in os.listdir("Roy/images_first_try/"):
    os.remove("Roy/images_first_try/"+image)


warnings.filterwarnings("ignore")
zoom = 3
path_to_folder = "Roy/images_first_try/"



options = selenium.webdriver.ChromeOptions()
#options.add_argument("--headless")   # run the browser in the background

driver = selenium.webdriver.Chrome(options=options)
url = "https://www.instantstreetview.com/@47.3768866,8.541694,0h,0p,1z"
driver.get(url)
driver.set_window_size(1920, 1080)
buttons = driver.find_elements(By.ID, "save")
print(buttons)
time.sleep(10)
buttons.click() 


for i in track(range(10)):
    # generate random latitude and longitude within street view limits
    lat = np.random.uniform(-70,80)
    lon = np.random.uniform(-180,180)
    
    # rule out China
    if lat >29 and lat <42 and lon > 85 and lon < 120:
        print("China")
        continue

    
    # check if the coordinates are on land
    if not (globe.is_land(lat, lon)):
        print("Not on land")
        continue
    print(lat, lon)

    # get the panoid from the coordinates
    url = "https://www.instantstreetview.com/@"+str(lat)+","+str(lon)+",0h,0p,1z"   
    #print("Opening URL" + url)
    # wait for the page to load
    driver.get(url)
    driver.implicitly_wait(5)
    time.sleep(5)

    print("Waiting for the page to load")



    # click the button that says "Alle ablehnen"
    current=driver.current_url
    if not current.__contains__("streetviewpixels-pa.googleapis"):
        print("Could not find street view")

        continue
    # get the panoid
    panoid = current.split("panoid%3D")[1]

    panoid = panoid.split("%")[0]
    print(panoid)




    for x in range(2**zoom):
        for y in range(2**(zoom-1)):
            if y == 0 or y == 2**(zoom-1)-1:
                #print(y)
                continue
            url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
            
            #check if the image exists
            response = requests.get(url)
            status_code = response.status_code
            
            save_path = path_to_folder+str(lat)+"_"+str(lon)+"_Index_"+str(x)+"_"+str(y)+".png"
            if status_code == 200:
                #print(panoid)
                print("Image exists")
            else:
                print("Image does not exist")
                # save a blank image
                img = Image.new("RGB", (512, 512))
                img.save(save_path)
                continue
                
            
            # open the link using Chrome
            options = selenium.webdriver.ChromeOptions()
            options.add_argument("--headless")   # run the browser in the background
            driver = selenium.webdriver.Chrome(options=options)
            driver.get(url)
            
            buttons = driver.find_elements(By.CSS_SELECTOR, "button")
            # delay to load the page
            time.sleep(0.5)
            
            
            #print(save_path)
            # save the image via screenshot
            driver.save_screenshot(save_path)
        
driver.quit()

print("zoom", zoom)
# Since the images have a massive black border around them, we need to crop them
# to the actual street view image
for image in os.listdir(path_to_folder):
    img = Image.open(path_to_folder+image)
    width, height = img.size
    left = width/2 -256
    top = height/2 -256
    right = width/2 +256
    bottom = height/2 +256
    
    
    img = img.crop((left, top, right, bottom))
    img.save(path_to_folder+image)


path_to_combined_folder = "Roy/combined_images/"
# combine 4 images into 1
for image in os.listdir(path_to_folder):
    img = image.split("_",3)
    ind = img[-1]
    ind = ind[0]
    img = img[0]+"_"+img[1]+"_"+img[2]+"_"
    print(img)
    print(ind)
    new_image = Image.new("RGB", (1024, 1024))
    x = int(ind)
    for y in [1,2]:
            #check if the image exists
            #default image as a black image
            image = Image.new("RGB", (512, 512))
            if os.path.exists(path_to_folder+img+str(x)+"_"+str(y)+".png"):
                image = Image.open(path_to_folder+img+str(x)+"_"+str(y)+".png")
            new_image.paste(image, (0, (y-1)*512))
            
            # handle wraparound
            if x == 2**zoom-1:
                x=-1
            
                
            image = Image.new("RGB", (512, 512))
            if os.path.exists(path_to_folder+img+str(x+1)+"_"+str(y)+".png"):
                image = Image.open(path_to_folder+img+str(x+1)+"_"+str(y)+".png")
            new_image.paste(image, (512, (y-1)*512))
            
            if x == -1:
                x = 2**zoom-1
    
    x+=1
    if x > 2**zoom-1:
        x = x - 2**zoom
    
    image = img
    new_image.save(path_to_combined_folder+image+"_Index_"+str(x+1)+".png")


print("Done")
    