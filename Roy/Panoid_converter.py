import requests 
import numpy as np
import os
import selenium
from selenium.webdriver.common.by import By
import json
import urllib.request as urllib
import re
from rich.progress import track


# generate random latitude and longitude
latitude = np.random.uniform(45,47)
longitude = np.random.uniform(8, 9)
zoom = 5
print(latitude, longitude)

MAPS_PREVIEW_ID = "CAEIBAgFCAYgAQ"
UNKNOWN_PREVIEW_CONSTANT = 45.12133303837374



panoid = "bl3v0-ol5SonuMF_4ozgxQ"

#now generate a similar panoid



for i in track(range(1000)):
    panoid = np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_-"), size=22)
    panoid = "".join(panoid)
    
    for x in range(3):
        url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid="+str(panoid)+"&x="+str(x)+"&y=0&zoom=5&nbt=1&fover=2"
        
        #check if the image exists
        response = requests.get(url)
        status_code = response.status_code
        if status_code == 200:
            print(panoid)
            print("Image exists")
        else:
            #print("Image does not exist")
            break
        
        # open the link using Chrome
        driver = selenium.webdriver.Chrome()
        driver.get(url)
        driver.maximize_window()
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        # delay to load the page
        driver.implicitly_wait(5)
    
    