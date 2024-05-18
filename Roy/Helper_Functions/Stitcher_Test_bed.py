import os
from PIL import Image
from rich.progress import track

# if there are not 1 and 2 images, remove it
for image in track(os.listdir("Roy/images_first_try/")):
    img = str(image).split("_",3)
    ind = img[-1]
    ind = ind[0]
    img = img[0]+"_"+img[1]+"_"+img[2]+"_"
    if not os.path.exists("Roy/images_first_try/"+img+str(ind)+"_1.png") or not os.path.exists("Roy/images_first_try/"+img+str(ind)+"_2.png"):
        print("Removing", image)
        os.remove("Roy/images_first_try/"+image)


zoom =3
path_to_folder = "Roy/images_first_try/"
path_to_combined_folder = "Roy/combined_images/"
for image in track(os.listdir(path_to_folder)):
    #print(image)
    img = Image.open(path_to_folder+image)
    if img.size[0] > 512 and img.size[1] > 512:
        #print("Cropping", image)
        width, height = img.size
        left = width/2 -256
        top = height/2 -256
        right = width/2 +256
        bottom = height/2 +256
    elif img.size[0] == 512 and img.size[1] == 512:
        continue
    
    else:
        print("Removing", image)
        os.remove(path_to_folder+image)
        continue
        
    
    
    img = img.crop((left, top, right, bottom))
    img.save(path_to_folder+image)


path_to_combined_folder = "Roy/combined_images/"
# combine 4 images into 1
for image in track(os.listdir(path_to_folder)):
    img = image.split("_",3)
    ind = img[-1]
    ind = ind[0]
    img = img[0]+"_"+img[1]+"_"+img[2]+"_"
    #print(img)
    #print(ind)
    new_image = Image.new("RGB", (1024, 1024))
    x = int(ind)
    for y in [1,2]:
            #check if the image exists
            #default image as a black image
            image = Image.new("RGB", (512, 512))
            if os.path.exists(path_to_folder+img+str(x)+"_"+str(y)+".png"):
                image = Image.open(path_to_folder+img+str(x)+"_"+str(y)+".png")
            new_image.paste(image, (0, (y-1)*512))
            
            # handle wraparound
            if x == 2**zoom-1:
                x=-1
            
                
            image = Image.new("RGB", (512, 512))
            if os.path.exists(path_to_folder+img+str(x+1)+"_"+str(y)+".png"):
                image = Image.open(path_to_folder+img+str(x+1)+"_"+str(y)+".png")
            new_image.paste(image, (512, (y-1)*512))
            
            if x == -1:
                x = 2**zoom-1
    
    x+=1
    if x > 2**zoom-1:
        x = x - 2**zoom
    
    image = img
    new_image.save(path_to_combined_folder+image+"_Index_"+str(x+1)+".png")


print("Done")
    