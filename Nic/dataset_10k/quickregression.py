import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.realpath(__file__))
csv_file_path = os.path.join(dir_path, 'coords.csv')

class GeoDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file, names=['x-coordinate', 'y-coordinate'])
        self.image_paths = [os.path.join(dir_path, 'dataset', f'{i}.jpg') for i in range(1, len(self.dataframe) + 1)]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        coordinates = self.dataframe.iloc[idx].values
        return idx+1, coordinates, image_path

# Create the dataset
dataset = GeoDataset(csv_file_path)
index, coordinates, image_path = dataset[0]
print(f'Index: {index}')
print(f'Coordinates: {coordinates}')
print(f'Image path: {image_path}')

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)