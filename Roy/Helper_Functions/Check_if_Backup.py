# Check if the data in D:/GeoGuessrGuessr/geoguesst is the same as on E:/geoguesst


import os


def check_if_backup():
    # load the names of the images in other folder different drive
    images = os.listdir("D:/GeoGuessrGuessr/geoguesst")
    images2 = os.listdir("E:/geoguesst_external")
    images.sort()
    images2.sort()
    if images == images2:
        print("The data is the same")
    else:
        print("The data is not the same")
        print("The data in D:/GeoGuessrGuessr/geoguesst is", len(images), "long")
        print("The data in E:/geoguesst_external is", len(images2), "long")
        
check_if_backup()