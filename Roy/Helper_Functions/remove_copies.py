import os
from PIL import Image
from rich.progress import track

def remove_copies():
    for image in track(os.listdir("E:/geoguesst_external")):
        if str(image).__contains__("Copy"):
            print("Removing", image)
            os.remove("E:/geoguesst_external/"+image)

remove_copies()