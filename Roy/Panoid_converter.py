import requests 
import numpy as np
import os
import selenium
from PIL import Image
from selenium.webdriver.common.by import By
import json
import urllib.request as urllib
import re
from rich.progress import track

zoom =3
path_to_folder = "Roy/images_first_try/"
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
            image = Image.open(path_to_folder+img+str(x)+"_"+str(y)+".png")
            new_image.paste(image, (0, (y-1)*512))
            # handle wraparound
            x= x+1
            if x == 2**zoom:
                x=0
            image = Image.open(path_to_folder+img+str(x)+"_"+str(y)+".png")
            new_image.paste(image, (512, (y-1)*512))
        
        
    image = img
    new_image.save(path_to_combined_folder+image+"_Index_"+str(x)+".png")


print("Done")
    