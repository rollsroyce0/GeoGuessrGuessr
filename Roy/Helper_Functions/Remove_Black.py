import os
from PIL import Image   
from rich.progress import track
#remove all fully black images from combined_images
counter = 0
for image in track(os.listdir("D:/GeoGuessrGuessr/geoguesst")):
    if image.endswith(".png"):
        img = Image.open("D:/GeoGuessrGuessr/geoguesst/"+image)
        if img.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove("D:/GeoGuessrGuessr/geoguesst/"+image)
            
print("removed", counter, "images")