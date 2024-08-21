import os
from PIL import Image   
from rich.progress import track
import matplotlib.pyplot as plt
#remove all fully black images from combined_images

def remove_black_images(image, counter, folder):
    if image.endswith(".png"):
        img = Image.open(folder+image)
        #print(img.getextrema())
        if img.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove(folder+image)
    return counter

def remove_half_black_images(image, counter, folder):
    if image.endswith(".png"):
        img = Image.open(folder+image)
        
        #retain only the left half of the image
        img1 = img.crop((0, 0, img.width//2, img.height))
        img2 = img.crop((img.width//2, 0, img.width, img.height))
  
        if img1.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove(folder+image)
            return counter
        if img2.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove(folder+image)
    return counter


counter = 0
folder = "D:/GeoGuessrGuessr/geoguesst/"
for image in track(os.listdir(folder)):
    previous_counter = counter
    counter += remove_half_black_images(image, counter, folder)
    if previous_counter != counter:
        continue
    counter += remove_black_images(image, counter, folder)
    
            
print("removed", counter, "images")


# check for duplicates
import os

def check_duplicates(folder):
    files = os.listdir(folder)
    print(len(files))
    print(len(set(files)))
    print(len(files)-len(set(files)))

check_duplicates("D:/GeoGuessrGuessr/geoguesst/")