import os
from PIL import Image   
from rich.progress import track
import matplotlib.pyplot as plt
#remove all fully black images from combined_images

def remove_black_images(image, counter):
    if image.endswith(".png"):
        img = Image.open("D:/GeoGuessrGuessr/geoguesst/"+image)
        print(img.getextrema())
        if img.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove("D:/GeoGuessrGuessr/geoguesst/"+image)
    return counter

def remove_half_black_images(image, counter):
    if image.endswith(".png"):
        img = Image.open("D:/GeoGuessrGuessr/geoguesst/"+image)
        
        #retain only the left half of the image
        img1 = img.crop((0, 0, img.width//2, img.height))
        img2 = img.crop((img.width//2, 0, img.width, img.height))
  
        if img1.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove("D:/GeoGuessrGuessr/geoguesst/"+image)
            return counter
        if img2.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove("D:/GeoGuessrGuessr/geoguesst/"+image)
    return counter


counter = 0
for image in track(os.listdir("D:/GeoGuessrGuessr/geoguesst")):
    if image.endswith("2.png") or image.endswith("3.png") or image.endswith("4.png") or image.endswith("5.png") or image.endswith("6.png") or image.endswith("7.png"):
        continue
    counter = remove_half_black_images(image, counter)
    
            
print("removed", counter, "images")