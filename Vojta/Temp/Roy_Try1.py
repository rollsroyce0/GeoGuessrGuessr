import os
import threading
import tkinter as tk
from tkinter import font
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.io import decode_image
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchsummary import summary
import geopy.distance
import geopandas as gpd
from torchvision.io import read_file, decode_image

warnings.filterwarnings("ignore")

########################################
# Tkinter Popout Window for Val Loss  #
########################################

def create_loss_window():
    root = tk.Tk()
    root.title("Latest Validation Loss")
    large_font = font.Font(family="Helvetica", size=72, weight="bold")
    label = tk.Label(root, text="Epoch: N/A\nVal Loss: N/A\nLowest Loss: N/A", font=large_font, bg="white", fg="black")
    label.pack(padx=20, pady=20)
    return root, label

def update_loss_label(label, epoch, new_loss, min_val_loss):
    points = np.floor(5000 * np.exp(-1 * new_loss / 2000))
    label.config(text=f"Epoch: {epoch}\nVal Loss: {new_loss:.4f} km\nLowest Loss: {min_val_loss:.4f} km\nGeoguessr Points: {points}")
    label.update_idletasks()

def launch_loss_window(event):
    global loss_root, loss_label
    loss_root, loss_label = create_loss_window()
    event.set()
    loss_root.mainloop()

window_ready_event = threading.Event()
threading.Thread(target=launch_loss_window, args=(window_ready_event,), daemon=True).start()
window_ready_event.wait()

#######################################
# Dataset and Model Definition        #
#######################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StreetviewDataset(Dataset):
    def __init__(self, location, transform=None, max_samples=None):
        self.location = location
        self.transform = transform
        self.samples = [f for f in track(os.listdir(location)) if f.endswith(".png")]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = os.path.join(self.location, self.samples[index])
        img_tensor = decode_image(read_file(img_path)).float()
        if self.transform:
            img_tensor = self.transform(img_tensor)

        lat, lon = map(float, self.samples[index].split("_")[:2])
        target = torch.tensor([lat, lon], dtype=torch.float32)
        return img_tensor, target

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 12, 9, 2), nn.BatchNorm2d(12), nn.Dropout2d(0.3), nn.MaxPool2d(2),
            nn.Conv2d(12, 48, 7), nn.BatchNorm2d(48), nn.Dropout2d(0.3), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(37632, 256), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

########################################
# Loss Functions and Training Setup    #
########################################

def degrees_to_radians(deg):
    return deg * np.pi / 180

def haversine_loss(coords1, coords2):
    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [
        coords1[:, 0], coords1[:, 1], coords2[:, 0], coords2[:, 1]
    ])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    return (2 * 6371.01 * torch.arcsin(torch.sqrt(a))).mean()

#@torch.compile
def compiled_haversine_loss(c1, c2):
    return haversine_loss(c1, c2)

geo_predictor = GeoPredictorNN().to(device)
#summary(geo_predictor, (3, 256, 256))

location = "D:/GeoGuessrGuessr/geoguesst/"
if not os.path.exists(location):
    raise FileNotFoundError(f"Dataset location '{location}' does not exist. Please check the path.")
dataset = StreetviewDataset(location)
print(f"Dataset contains {len(dataset)} samples.")
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

batch_size = 2000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(f"Train loader batches: {len(train_loader)}, Test loader batches: {len(test_loader)}")
criterion = MSELoss()
optimizer = optim.AdamW(geo_predictor.parameters(), lr=1, weight_decay=1e-2, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.98)

print("Model and data loaders initialized successfully.")
########################################
# Training Loop                        #
########################################

epochs = 10
losses, val_losses = [], []
min_val_loss = float('inf')

print("Starting training...")

for epoch in range(epochs):
    geo_predictor.train()
    running_loss = 0.0
    for images, coords in track(train_loader, description="Training..."):
        optimizer.zero_grad()
        outputs = geo_predictor(v2.AutoAugment()(images))
        loss = criterion(outputs, coords)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss / len(train_loader))

    geo_predictor.eval()
    val_loss = sum(compiled_haversine_loss(geo_predictor(images), coords).item() for images, coords in test_loader) / len(test_loader)
    val_losses.append(val_loss)
    update_loss_label(loss_label, epoch + 1, val_loss, min_val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(geo_predictor.state_dict(), f'./Saved_Models/TMP/best_model.pth')

print("Training Complete.")
loss_root.quit()