from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import time
import os
import io
import base64
import re

# --- Config ---
SESSION_URL = "https://www.geoguessr.com/challenge/IulVIxZaUpOjrARD"
NUM_ROUNDS = 5  # Update based on number of rounds
SAVE_DIR = "geoguessr_screenshots"
CROP_BOX = (0, 120, 1920, 940)  # Adjust based on screen/UI layout

# --- Setup ---
os.makedirs(SAVE_DIR, exist_ok=True)
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

coordinates = []

def crop_and_save(img_data, filename):
    img = Image.open(io.BytesIO(img_data))
    cropped = img.crop(CROP_BOX)
    cropped.save(os.path.join(SAVE_DIR, filename))

# --- Main ---
driver.get(SESSION_URL)
# Wait for and click 'Accept Cookies' button
try:
    cookie_banner = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-qa='cookie-banner-accept']")))
    ActionChains(driver).move_to_element(cookie_banner).click().perform()
    print("Accepted cookies.")
except:
    print("No cookie banner found (possibly already accepted).")

# Enter nickname and click 'Play as guest'
try:
    nickname_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[data-qa='guest-nickname-input']")))
    nickname_input.send_keys("AutoPlayer")

    play_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-qa='start-guest-game-button']")))
    play_button.click()
    print("Entered nickname and started as guest.")

    # Wait for game layout to load
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "game-layout")))
    time.sleep(2)
except Exception as e:
    print(f"Guest login step failed: {e}")


for round_idx in range(NUM_ROUNDS):
    print(f"Processing round {round_idx + 1}")

    # Go to round if results page
    try:
        round_btns = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".result-layout__round")))
        round_btns[round_idx].click()
    except:
        print("Could not find round buttons.")
        break

    # Wait for panorama to load
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "game-layout")))

    time.sleep(2)
    
    # Screenshot (Base64 to bytes)
    screenshot = driver.get_screenshot_as_png()
    filename = f"round_{round_idx + 1}.png"
    crop_and_save(screenshot, filename)

    # Click Make Guess if not already guessed
    try:
        guess_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "guess-map__button")))
        guess_button.click()
        time.sleep(2)
    except:
        pass  # Already guessed

    # Click the result pin to reveal coordinates
    try:
        pin = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "result-layout__actual-location-marker")))
        pin.click()
        time.sleep(1)
    except:
        print("Couldn't click location pin.")
        coordinates.append("Unknown")
        continue

    # Extract coordinates from the popup
    try:
        coord_text = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "coordinate"))).text
        match = re.findall(r"(-?\d+\.\d+)", coord_text)
        lat, lon = match if len(match) == 2 else ("?", "?")
    except:
        lat, lon = "?", "?"

    coordinates.append((lat, lon))

# Save coordinates
with open(os.path.join(SAVE_DIR, "coordinates.txt"), "w") as f:
    for idx, (lat, lon) in enumerate(coordinates):
        f.write(f"Round {idx+1}: {lat}, {lon}\n")

driver.quit()
print("Done!")
