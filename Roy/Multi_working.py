# create a webscraper that will scrape Google maps streetview images for random locations and save the images to a folder and the coordinates to a csv file
# Do not use the Google Maps API, run the scraper in a way that will not get you banned with 
# import libraries
import requests
import numpy as np
import os
import csv
import random
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

# create a function that will scrape the Google Maps streetview images
def get_images():
    # generate random coordinates
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    base_orientation = np.random.uniform(0, 45)
    for i in range(8):
        orientation = base_orientation + i*45
        # create a url with the coordinates
        url = f"https://www.google.com/maps/@{lat},{lon},3a,75y,{orientation}h/data=!3m7!1e1!3m5!1sRs4wJ_bJ_tRQpj5XFly9kw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fpanoid%3DRs4wJ_bJ_tRQpj5XFly9kw%26cb_client%3Dmaps_sv.share%26w%3D900%26h%3D600%26yaw%3D315.9589397501515%26pitch%3D8.341132131762492%26thumbfov%3D90!7i16384!8i8192?coh=205410&entry=ttu"
        
        # open the link using firefox
        
        
        driver = webdriver.Chrome()
        driver.get(url)
        driver.maximize_window()
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        print(buttons)
        buttons[1].click()
        # wait for the page to load
        time.sleep(7)
        # click the button that says "Alle ablehnen"
        


# create main function
def main():
    # create a folder to save the images
    os.makedirs("datasets/images_first_try", exist_ok=True)
    # create a csv file to save the coordinates
    with open("Roy/coordinates.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["latitude", "longitude"])
    # create a loop to scrape the images
    for i in range(10):
        get_images()
        time.sleep(1)
        
        
# run the main function
main()