import requests
from PIL import Image
from rich.progress import track
import selenium
from selenium.webdriver.common.by import By
import os
import time
from selenium.webdriver.common.action_chains import ActionChains
import warnings

warnings.filterwarnings("ignore")


    
# generate random latitude and longitude
lat = 47.3667985
lon = 8.5430297

print(lat, lon)

# get the panoid from the coordinates
url = "https://www.google.ch/maps/@"+str(lat)+","+str(lon)+",17z?entry=ttu"

options = selenium.webdriver.ChromeOptions()
#options.add_argument("--headless")   # run the browser in the background

driver = selenium.webdriver.Chrome(options=options)

driver.get(url)
driver.set_window_size(1920, 1080)
buttons = driver.find_elements(By.CSS_SELECTOR, "button")
#print(buttons)
buttons[1].click()
# wait for the page to load
driver.refresh()
driver.implicitly_wait(5)
time.sleep(3)


buttons = driver.find_elements(By.CSS_SELECTOR, "button")

#time.sleep(70000)
while buttons.__len__() <27:
    
    driver.refresh()
    time.sleep(3)
    buttons = driver.find_elements(By.CSS_SELECTOR, "button")
    


print(buttons[26])

# drag the street view to a random location
element=buttons[26]
action = ActionChains(driver)
action.move_to_element(element).click_and_hold().move_by_offset(-800, -500).release().perform()


print("Waiting for the page to load")
time.sleep(3)
driver.implicitly_wait(5)


# click the button that says "Alle ablehnen"
current=driver.current_url
if not current.__contains__("streetviewpixels-pa.googleapis"):
    print("Not the correct page")
    driver.quit()
    exit()

driver.quit()
# get the panoid
panoid = current.split("panoid%3D")[1]
print(panoid)
panoid = panoid.split("%")[0]
print(panoid)



zoom = 3
path_to_folder = "Roy/images_first_try/"
for x in range(2**zoom):
    for y in range(2**(zoom-1)):
        if y == 0 or y == 2**(zoom-1)-1:
            continue
        url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid="+str(panoid)+"&x="+str(x)+"&y="+str(y)+"&zoom="+str(zoom)+"&nbt=1&fover=2"
        
        #check if the image exists
        response = requests.get(url)
        status_code = response.status_code
        if status_code == 200:
            #print(panoid)
            print("Image exists")
        else:
            print("Image does not exist")
            break
        
        # open the link using Chrome
        options = selenium.webdriver.ChromeOptions()
        options.add_argument("--headless")   # run the browser in the background
        driver = selenium.webdriver.Chrome(options=options)
        driver.get(url)
        
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        # delay to load the page
        time.sleep(1)
        
        save_path = path_to_folder+str(lat)+"_"+str(lon)+"_Index_"+str(x)+"_"+str(y)+".png"
        #print(save_path)
        # save the image via screenshot
        driver.save_screenshot(save_path)
        
driver.quit()

print("zoom", zoom)
# Since the images have a massive black border around them, we need to crop them
# to the actual street view image
for image in os.listdir(path_to_folder):
    img = Image.open(path_to_folder+image)
    width, height = img.size
    left = width/2 -256
    top = height/2 -256
    right = width/2 +256
    bottom = height/2 +256
    
    
    img = img.crop((left, top, right, bottom))
    img.save(path_to_folder+image)


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
    