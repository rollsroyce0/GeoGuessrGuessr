import requests 
import numpy as np
import os
import selenium
from selenium.webdriver.common.by import By
import json
import re


MAPS_PREVIEW_ID = "CAEIBAgFCAYgAQ"
UNKNOWN_PREVIEW_CONSTANT = 45.12133303837374
latitude = np.random.uniform(45,47)
longitude = np.random.uniform(8, 9)
zoom = 2
print(latitude, longitude)

client_id = requests.get(url="https://www.google.com/maps").json()
print(client_id)


preview_document = json.loads(
    requests.get(
        url="https://www.google.com/maps/preview/photo?authuser=0&hl=en&gl=us&pb=!1e3!5m54!2m2!1i203!2i100!3m3!2i4!3s%s!5b1!7m42!1m3!1e1!2b0!3e3!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e9!2b1!3e2!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e4!2b1!4b1!8m0!9b0!11m1!4b1!6m3!1s%s!7e81!15i11021!9m2!2d%f!3d%f!10d%f"
        % (MAPS_PREVIEW_ID, client_id, longitude, latitude, UNKNOWN_PREVIEW_CONSTANT)
    ).text[4:]
)



for x in range(4):
    url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid=bl3v0-ol5SonuMF_4ozgxQ&x="+str(x)+"&y=0&zoom=2&nbt=1&fover=2"
    
    # open the link using Chrome
    driver = selenium.webdriver.Chrome()
    driver.get(url)
    driver.maximize_window()
    buttons = driver.find_elements(By.CSS_SELECTOR, "button")
    
    