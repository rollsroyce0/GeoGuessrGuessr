import re, time, random
from playwright.sync_api import sync_playwright
from PIL import Image
#from ML.Second_Level_ML.generate_coordinates import list_test_types
import numpy as np
import os

ROUND_COUNT = 5
WAIT_SV = 2.2

def extract_coords(url):
    m = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', url)
    return tuple(map(float, m.groups())) if m else None

def list_test_types():
    with open('Roy/ML/Second_Level_ML/generate_coordinates.py', 'r') as f:
        content = f.read()
    matches = re.findall(r"'([A-Za-z]+)'", re.search(r'def list_test_types\(\):\s*return \[(.*?)\]', content, re.DOTALL).group(1))
    return list(matches)

def run(geoguessr_url, used_types):
    coords = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        #fullscreen
        page.set_viewport_size({"width": 2560, "height": 1440})

        page.goto(geoguessr_url)
        page.wait_for_timeout(400)

        #accept cookies
        try:
            page.get_by_text("Accept all & visit the site").click()
            print("Accepted cookies.")
        except:
            print("No cookies prompt.")
        page.wait_for_timeout(2000)
        print("entering Nickname...")
        #enter nickname
        try:
            page.get_by_placeholder("Nickname").click()
            page.keyboard.type("MLTester")
            page.get_by_text("Play as Guest").click()
            print("Entered nickname and started game.")
        except:
            print("No nickname prompt.")
        #------------------------------------------------------------------------------
        # screenshot dimensions: 59 pixels from top start, all the way to bottom, 170 from left, to 450 pixels from right
        #------------------------------------------------------------------------------
        
        for i in range(ROUND_COUNT):
            page.wait_for_timeout(1500)
            print(f'Round {i+1}')
            
            #take screenshot of current view
            page.screenshot(path=f'Roy/Helper_Functions/Automated_Validation_images/screenshots_temp/round_{i+1}_start.jpg', full_page=True)
            time.sleep(0.5)
            #open the screenshot and crop the top bar

            img = Image.open(f'Roy/Helper_Functions/Automated_Validation_images/screenshots_temp/round_{i+1}_start.jpg')
            width, height = img.size
            print(f'  Original screenshot size: {width}x{height}')
            cropped_img = img.crop((170, 59, width - 450, height))
            cropped_img.save(f'Roy/Helper_Functions/Automated_Validation_images/screenshots_temp/round_{i+1}.jpg')
            print('  Screenshot taken.')

            page.wait_for_timeout(100)
            page.mouse.click(2300, 1300)
            print('  Clicked to focus map.')
            page.wait_for_timeout(1000)

            # click guess button
            page.get_by_text("Guess").click()
            print('  Clicked Guess button.')
            page.wait_for_timeout(1000)
            
            
            # click flag (opens Street View)
            page.get_by_alt_text("Correct location").click()
            print('  Clicked flag to open Street View.')

            # wait for new tab
            sv = ctx.wait_for_event("page")
            sv.wait_for_timeout(1000)
            sv.set_viewport_size({"width": 2560, "height": 1440})
            
            # works up to here
            
            #accept google cookies if the aria-label button exists
            if sv.is_visible("[aria-label='Accept all']"):
                try:
                    sv.click("[aria-label='Accept all']")
                except:
                    time.sleep(0.05)
            sv.wait_for_timeout(4000)
            print("  Accepted Google cookies if prompted.")
            url = sv.url
            print(f'  Street View URL: {url}')
            time.sleep(0.5)
            c = extract_coords(url)
            if c is None:
                print('  Failed to extract coordinates!')
                # try again after waiting
                sv.wait_for_timeout(2000)
                url = sv.url
                c = extract_coords(url)
                if c is None:
                    print('  Second attempt failed, skipping this round.')
            
            coords.append(c)

            print(f'  Extracted coordinates: {c}')
            
            sv.close()
            print('  Closed Street View tab.')
            
            #time.sleep(1000)
            # next round
            if i < ROUND_COUNT - 1:
                page.keyboard.press("Space")
                print('  Clicked Next round button.')
                page.wait_for_timeout(2000)

        browser.close()
        
    

    print('\nFinal coordinates:')
    for i, c in enumerate(coords, 1):
        print(f'{i}: {c}')
        
    # real_coords_Yippee = np.array([[37.2461187,-76.6488819], [17.5421013,80.6055166], [-29.3102639,27.5317762], [50.7016383,20.653696], [30.9914195,-97.82943]])  put it into this format, with a 4 letter name. check wether its used already from the list_test_types function in generate_coordinates.py
    used_test_types = list_test_types()
    print(used_test_types)
    used_types = list(used_types)
    print(used_types)
    
    if len(used_types)>0:
        print("test")
        for element in used_types:
            if not used_test_types.__contains__(element):
                used_test_types.append(element)
    print(f'\nUsed test types: {used_test_types}')
    
    # Generate a random test type name that is not already used out of 4 characters, then check against the first 4 letters of each used
    while True:
        test_type = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
        if test_type not in [name[:4] for name in used_test_types]:
            break
    print(f'\nSuggested test type name: {test_type}')
    used_types.append(str(test_type))
    

    print('\nIn array format:')
    array_str = 'real_coords_'+test_type+' = np.array(['
    for c in coords:
        array_str += f'[{c[0]},{c[1]}], '

    array_str = array_str.rstrip(', ') + '])'
    print(array_str)
    

    # take the test_type, and rename all files as such: XXXX_Test1.jpg, XXXX_Test2.jpg, etc
    for i in range(ROUND_COUNT):
        old_path = f'Roy/Helper_Functions/Automated_Validation_images/screenshots_temp/round_{i+1}.jpg'
        new_path = f'Roy/Test_Images/{test_type}_Test{i+1}.jpg'
        img = Image.open(old_path)
        img.save(new_path)
        img.close()

    with open('Roy/Helper_Functions/Automated_Validation_images/screenshots_temp/temp_coords.txt', 'w') as f:
        f.write(array_str)
        
    with open('Roy/ML/Second_Level_ML/generate_coordinates.py', 'r+') as f:
        lines = f.readlines()
        #print(lines)
        insert_index_array = None
        insert_index_ifelse = None
        insert_index_testtypes = None
        for i, line in enumerate(lines):
            #print(f'Line {i}: {line.strip()}')
            if line.strip() == "# New arrays below here":
                insert_index_array = i + 1
                continue
            if line.strip() == "#new test types added here":
                insert_index_testtypes = i + 4
                continue
            if line.strip() == "# New If-Else below here":
                insert_index_ifelse = i + 2
                continue
            
        print("\nModifying generate_coordinates.py...")
        print(f'Found insert indices - Array: {insert_index_array}, If-Else: {insert_index_ifelse}, Test Types: {insert_index_testtypes}')
        if insert_index_array is not None:
            print(f'Inserting new array at line {insert_index_array}')
            lines.insert(insert_index_array, "    "+ array_str+"\n")
            
        if insert_index_ifelse is not None:
            print(f'Inserting new if-else at line {insert_index_ifelse}')
            lines.insert(insert_index_ifelse, "    elif testtype == '"+test_type+"':"+"\n")
            lines.insert(insert_index_ifelse+1, "        real_coords = real_coords_"+test_type+"\n")
            
        if insert_index_testtypes is not None:
            print(f'Inserting new test type at line {insert_index_testtypes}')
            lines.insert(insert_index_testtypes, "        '"+test_type+"',"+"\n")
        
        f.seek(0)
        f.truncate()
        f.writelines(lines)
    
    return used_types
        
        

if __name__ == "__main__":
    List_of_links = [
        "https://www.geoguessr.com/challenge/gsGEAMcC29zDXPE7",
        "https://www.geoguessr.com/challenge/K3y3S4WkFieTwnlq",
        "https://www.geoguessr.com/challenge/XGawCvuu52N4TXra",
        "https://www.geoguessr.com/challenge/SnZQSfahHXSdGMtG",
        "https://www.geoguessr.com/challenge/LdKG7vAtdPDMIe7T"
    ]
    
    
    used_types = []
    for i in range (5):
        used_types = run(List_of_links[i], used_types)
