import os
from PIL import Image   
from rich.progress import track
#remove all fully black images from combined_images
counter = 0
for image in track(os.listdir("Roy/combined_images/")):
    if image.endswith(".png"):
        img = Image.open("Roy/combined_images/"+image)
        if img.getextrema() == ((0,0), (0,0), (0,0)):
            print("removing", image)
            counter +=1
            os.remove("Roy/combined_images/"+image)
            
print("removed", counter, "images")