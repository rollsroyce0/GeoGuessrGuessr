import requests
import sys
from PIL import Image
import matplotlib.pyplot as plt
from rich.progress import track
import selenium
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import os
import numpy as np
import time
import datetime
import logging
from global_land_mask import globe
from streetview import search_panoramas
import warnings
sys.path.append('../')
import geographic_utils as geofindurban

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('geoscrape_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.realpath(__file__))
images_dir = os.path.join(script_dir, 'images_first_try')

# Clear existing images with debug logging
logger.info(f"Clearing existing images from directory: {images_dir}")
image_count = 0
for image in os.listdir(images_dir):
    try:
        os.remove(images_dir+'\\'+image)
        image_count += 1
    except Exception as e:
        logger.warning(f"Failed to remove image {image}: {e}")

logger.info(f"Removed {image_count} existing images")

warnings.filterwarnings("ignore")
zoom = 3
logger.info(f"Configuration: zoom level = {zoom}, images directory = {images_dir}")

options = selenium.webdriver.ChromeOptions()
options.add_argument('log-level=3')
options.add_argument("--headless")
logger.info("Chrome options configured for headless mode")

def generate_random_coords():
    """Generate random coordinates in urban areas with debug logging"""
    logger.debug("Generating random urban coordinates...")
    try:
        lat, lon = geofindurban.generate_random_point_in_urban_area()
        logger.info(f"Generated coordinates: lat={lat:.6f}, lon={lon:.6f}")
        return lat, lon
    except Exception as e:
        logger.error(f"Failed to generate random coordinates: {e}")
        raise

def find_panoid(lat, lon):
    """Find Street View panoid for given coordinates with comprehensive debug logging"""
    logger.debug(f"Searching for panoid at lat={lat:.6f}, lon={lon:.6f}")

    try:
        start_time = time.time()
        panoids = search_panoramas(lat=lat, lon=lon)
        search_duration = time.time() - start_time

        logger.debug(f"Street view search completed in {search_duration:.3f} seconds")
        logger.debug(f"Found {len(panoids)} panoids")

        if len(panoids) != 0:
            try:
                panoid = str(panoids[0]).split("'")[1].split("'")[0]
                logger.info(f"Panoid found: {panoid}")
                return panoid
            except Exception as parse_error:
                logger.error(f"Failed to parse panoid from response {panoids[0]}: {parse_error}")
                return None
        else:
            logger.warning(f"No panoids found at lat={lat:.6f}, lon={lon:.6f}")
            return None

    except Exception as e:
        logger.error(f"Error searching for panoid at lat={lat:.6f}, lon={lon:.6f}: {e}")
        return None

def get_images(driver, lat, lon, panoid, zoom):
    """Download and process Street View images with detailed progress logging"""
    logger.info(f"Starting image download for panoid {panoid} at lat={lat:.6f}, lon={lon:.6f}")

    total_images = 2**zoom * 2  # 2 y values per x
    image_count = 0
    success_count = 0
    start_time = time.time()

    for x in range(2**zoom):
        for y in [1,2]:
            image_count += 1
            logger.debug(f"Processing image {image_count}/{total_images}: x={x}, y={y}")

            try:
                # Construct URL and download image
                url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
                logger.debug(f"Requesting URL: {url}")

                response = requests.get(url)
                status_code = response.status_code
                logger.debug(f"HTTP Status: {status_code}")

                save_path = images_dir+'\\'+str(lat)+"_"+str(lon)+"_Index_"+str(x)+"_"+str(y)+".png"
                logger.debug(f"Save path: {save_path}")

                # Handle different response scenarios
                if status_code == 200:
                    if x == 0 and y == 1:
                        logger.info("First image downloaded successfully")
                    else:
                        logger.debug("Image downloaded successfully")
                else:
                    logger.warning(f"Non-200 status code {status_code} for image x={x}, y={y}. Creating blank image.")
                    img = Image.new("RGB", (512, 512))
                    img.save(save_path)

                # Use Selenium to get screenshot
                logger.debug("Loading URL in browser...")
                driver.get(url)
                time.sleep(0.1)

                logger.debug("Taking screenshot...")
                driver.save_screenshot(save_path)

                # Crop the image
                logger.debug("Cropping image...")
                img = Image.open(save_path)
                width, height = img.size
                left = width/2 - 256
                top = height/2 - 256
                right = width/2 + 256
                bottom = height/2 + 256

                img_cropped = img.crop((left, top, right, bottom))
                img_cropped.save(save_path)
                logger.debug(f"Successfully saved cropped image: {save_path}")
                success_count += 1

            except Exception as e:
                logger.error(f"Failed to process image x={x}, y={y}: {e}")
                # Create blank image as fallback
                try:
                    img = Image.new("RGB", (512, 512))
                    img.save(save_path)
                    logger.warning(f"Created blank fallback image for x={x}, y={y}")
                except Exception as fallback_error:
                    logger.error(f"Failed to create fallback image: {fallback_error}")

    # Summary logging
    duration = time.time() - start_time
    logger.info(f"Completed image processing: {success_count}/{total_images} images successful in {duration:.2f} seconds")
    if success_count < total_images:
        logger.warning(f"Only {success_count}/{total_images} images processed successfully")

    print(".", end="")  # Keep original minimal console output

def main():
    """Main function with comprehensive startup, processing, and shutdown logging"""
    logger.info("=== Starting GeoScrape2 ===")
    logger.info(f"Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {script_dir}")
    logger.info(f"Images will be saved to: {images_dir}")

    try:
        # Initialize WebDriver
        logger.info("Initializing Chrome WebDriver...")
        driver = selenium.webdriver.Chrome(options=options)

        # Navigate to Google Maps
        url = "https://www.google.ch/maps/"
        logger.info(f"Navigating to: {url}")
        driver.get(url)
        driver.set_window_size(1920, 1080)
        logger.info("Browser window set to 1920x1080")

        # Handle cookie banner
        logger.info("Attempting to handle cookie banner...")
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")

        try:
            # Wait for the cookie banner to be present and clickable
            cookie_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-qa='cookie-banner-accept']"))
            )
            ActionChains(driver).move_to_element(cookie_button).click().perform()
            logger.info("Successfully accepted cookies using data-qa selector")
        except Exception as cookie_error:
            logger.warning(f"Could not accept cookies using data-qa selector: {cookie_error}")
            # Fallback: try the original approach
            try:
                buttons = driver.find_elements(By.CSS_SELECTOR, "button")
                if len(buttons) > 1:
                    ActionChains(driver).move_to_element(buttons[1]).click().perform()
                    logger.info("Successfully accepted cookies using fallback method")
                else:
                    logger.warning("No cookie buttons found - proceeding without cookie acceptance")
            except Exception as fallback_error:
                logger.error(f"Could not accept cookies using fallback method: {fallback_error}")

        # Main processing loop
        logger.info("Starting main processing loop...")
        total_attempts = 0
        successful_locations = 0

        for iteration in track(range(3), description="Processing..."):
            logger.info(f"=== Starting iteration {iteration + 1}/3 ===")
            panoid = None
            attempts = 0

            # Find valid panoid
            logger.info("Searching for valid Street View location...")
            while panoid is None:
                try:
                    lat, lon = generate_random_coords()
                    panoid = find_panoid(lat, lon)
                    attempts += 1
                    total_attempts += 1

                    if attempts % 10 == 0:
                        logger.info(f"Attempt {attempts}: Still searching for valid panoid...")

                except Exception as e:
                    logger.error(f"Error during coordinate generation/search: {e}")
                    attempts += 1
                    total_attempts += 1
                    time.sleep(1)  # Brief pause to avoid overwhelming the system

            logger.info(f"Found valid panoid({panoid}) after {attempts} attempts")
            print(attempts)  # Keep original console output

            # Process images for this location
            try:
                get_images(driver, lat, lon, panoid, zoom)
                successful_locations += 1
                logger.info(f"Successfully processed location {successful_locations}/3")
            except Exception as e:
                logger.error(f"Failed to process images for location: {e}")

        # Cleanup
        logger.info("Processing complete. Quitting WebDriver...")
        driver.quit()
        logger.info("WebDriver quit successfully")

        # Summary statistics
        logger.info("=== Processing Summary ===")
        logger.info(f"Total iterations: 3")
        logger.info(f"Successful locations: {successful_locations}")
        logger.info(f"Total coordinate attempts: {total_attempts}")
        logger.info(f"Average attempts per location: {total_attempts/3:.1f}")
        logger.info(f"Script completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=== GeoScrape2 Completed ===")

    except Exception as main_error:
        logger.critical(f"Fatal error in main function: {main_error}")
        try:
            driver.quit()
            logger.info("WebDriver cleaned up after error")
        except:
            logger.error("Failed to clean up WebDriver after error")
        raise

if __name__ == "__main__":
    main()