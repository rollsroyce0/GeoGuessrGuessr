import os
import threading
import tkinter as tk
from tkinter import font
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import geopy.distance
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import geopandas as gpd
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

#######################################
# Utility: Extract Coordinates        #
#######################################
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon



location = "D:/GeoGuessrGuessr/geoguesst"

#######################################
# Define Transformations                #
#######################################
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


##########################
# Load or Generate Embeddings           #
#######################################
AVGCOL_file = 'Roy/ML/Avg_Color_embeddings.npy'
if os.path.exists(AVGCOL_file):
    print("Loading precomputed Avg Color embeddings...")
    avg_color_embeddings = np.load(AVGCOL_file)
else:
    print("Generating Avg Color embeddings...")
    image_paths = [os.path.join(location, f) for f in os.listdir(location) if f.endswith('.png')]
    avg_color_embeddings = np.zeros((len(image_paths), 3), dtype=np.float32)
    
    for i, image_path in enumerate(track(image_paths, description="Processing images...")):
        image = Image.open(image_path).convert('RGB')
        avg_color = np.array(image).mean(axis=(0, 1))
        avg_color_embeddings[i] = avg_color / 255.0
    
    # save the embeddings to a file as well as their paths
    image_paths = np.array(image_paths)
    np.save('Roy/ML/AVGCOL_image_paths.npy', image_paths)
    np.save(AVGCOL_file, avg_color_embeddings)
print(f"first few embeddings: {avg_color_embeddings[:5]}")
print("Avg Color embeddings saved to file.")

# Check if AVGCOL_image_paths.npy is the same as image_paths.npy
image_paths_file = 'Roy/ML/image_paths.npy'
if os.path.exists(image_paths_file):
    image_paths_loaded = np.load(image_paths_file)
    if np.array_equal(image_paths_loaded, image_paths):
        print("image_paths.npy is the same as AVGCOL_image_paths.npy")
    else:
        print("image_paths.npy is different from AVGCOL_image_paths.npy")
    
    