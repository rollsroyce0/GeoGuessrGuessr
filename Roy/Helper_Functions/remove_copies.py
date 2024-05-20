import os
from PIL import Image
from rich.progress import track

def remove_copies():
    for image in track(os.listdir("D:/GeoGuessrGuessr/geoguesst")):
        if str(image).__contains__("Copy"):
            print("Removing", image)
            os.remove("D:/GeoGuessrGuessr/geoguesst/"+image)

remove_copies()