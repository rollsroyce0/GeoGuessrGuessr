import os
import threading
import tkinter as tk
from tkinter import font
import time
from torchvision.io import read_file, decode_image

from shapely.geometry import geo
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torch.nn import L1Loss, MSELoss
from torchvision import models, transforms
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchsummary import summary
import geopy.distance
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import geopandas as gpd
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

loss_label = None

#######################################
# Tkinter Popout Window for Val Loss  #
#######################################
def create_loss_window():
    global loss_label
    root = tk.Tk()
    root.title("Latest Validation Loss")
    large_font = font.Font(family="Helvetica", size=72, weight="bold")
    loss_label = tk.Label(root, text="Epoch: N/A\nVal Loss: N/A\nLowest Loss: N/A", font=large_font, bg="white", fg="black")
    loss_label.pack(padx=20, pady=20)
    return root, loss_label

def update_loss_label(loss_label, epoch, new_loss, min_val_loss):
    points = np.floor(5000 * np.exp(-1*new_loss/2000))
    loss_label.config(text=f"Epoch: {epoch}\nVal Loss: {new_loss:.4f} km\nLowest Loss: {min_val_loss:.4f} km\nGeoguessr Points: {points}")
    loss_label.update_idletasks()

def launch_loss_window(window_ready_event):
    global loss_root, loss_label
    loss_root, loss_label = create_loss_window()
    window_ready_event.set()  # signal that window is ready
    loss_root.mainloop()

# Start the Tkinter window in a separate thread
window_ready_event = threading.Event()
threading.Thread(target=launch_loss_window, args=(window_ready_event,), daemon=True).start()
window_ready_event.wait()  # Wait until the window is ready


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StreetviewDataset(Dataset):
    def __init__(self, location):
        self.location = location
        self.image_paths = [img_file for img_file in os.listdir(self.location)]
        self.coordinates = torch.stack([self.extract_coordinates(path) for path in self.image_paths])

        #self.transform = transforms.Compose([
            #transforms.Resize((64, 64)),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            # std=[0.229, 0.224, 0.225])
        #])

        #self.images = torch.stack([self.transform(decode_image(read_file(self.location + self.image_paths[idx])).float()) for idx in track(range(len(self.coordinates)), description="Loading images... ")])
        #torch.save(self.images, "D:/GeoGuessrGuessr/TEMP/all_images_256x256.pt")
        #torch.save(self.coordinates, "D:/GeoGuessrGuessr/TEMP/all_coords_256x256.pt")

        self.images = torch.load("D:/GeoGuessrGuessr/TEMP/all_images_256x256.pt").pin_memory()
        self.coordinates = torch.load("D:/GeoGuessrGuessr/TEMP/all_coords_256x256.pt")

    # Function to extract coordinates from image path
    def extract_coordinates(self, image_path):
        
        coords = image_path.split("_")
        lat = float(coords[0])
        lon = float(coords[1])
        return torch.tensor([lat, lon]).to(device)
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx].to(device, non_blocking=True), self.coordinates[idx]

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Four 3×3 conv blocks, doubling channels each time
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 64×64 → 32×32

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32×32 → 16×16

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16×16 → 8×8

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8×8 → 4×4
        )

        # A little dropout to regularize
        self.dropout = nn.Dropout(0.3)

        # Two fully-connected layers for regression to (lat, lon)
        # 256 channels × 4×4 spatial → 4096 features
        self.regressor = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)                     # → [B,256,4,4]
        x = x.view(x.size(0), -1)                # → [B, 256*4*4]
        x = self.dropout(x)
        x = self.regressor(x)                    # → [B,2]
        return x

#######################################
# Define Haversine Loss and Optimizer   #
#######################################
def degrees_to_radians(deg):
    return deg * 0.017453292519943295

def haversine_distance(coords1, coords2):
    return geopy.distance.geodesic(coords1, coords2).kilometers

def haversine_loss(coords1, coords2):
    lat1, lon1 = coords1[:, 0], coords1[:, 1]
    lat2, lon2 = coords2[:, 0], coords2[:, 1]
    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    distance = 6371.01 * c
    return distance.mean()

def geo_loss(coords1, coords2):
    return torch.exp(-haversine_loss(coords1, coords2))

def evaluate_nn_model_light(geo_predictor):

    geo_predictor.eval()
    y_pred = np.array([[]]).reshape(0,2)
    y_test = np.array([[]]).reshape(0,2)
    distances = np.array([]).reshape(0)
    with torch.no_grad():
        for X, y in test_loader:
            predicted_coords = geo_predictor(X).cpu().numpy()

            predicted_coords[:, 0] = (predicted_coords[:, 0] + 90) % 180 - 90
            predicted_coords[:, 1] = (predicted_coords[:, 1] + 180) % 360 - 180

            y_pred = np.append(y_pred, predicted_coords, axis=0)
            y_test = np.append(y_test, y.cpu().numpy(), axis=0)

            distances = np.append(distances, np.array([haversine_distance(y[i].cpu().numpy(), predicted_coords[i]) for i in range(y.shape[0])]))


    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

    plt.figure(figsize=(10, 8))
    world = gpd.read_file(url)
    world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')

    plt.scatter(y_test[:100, 1], y_test[:100, 0], color='blue', label='True')
    plt.scatter(y_pred[:100, 1], y_pred[:100, 0], color='red', label='Predicted')
    for i in range(100):
        plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='green', alpha=0.5)
    plt.title('True vs Predicted Coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show(block =False); plt.pause(2)
    plt.close()

    return np.mean(distances), distances

# Initialize the predictor model
geo_predictor = GeoPredictorNN().to(device)
summary(geo_predictor, (3,64,64))
#geo_predictor.compile()

location = "D:/GeoGuessrGuessr/geoguesst/"
dataset = StreetviewDataset(location=location)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

#haversine_loss = torch.compile(haversine_loss)
#geo_loss = torch.compile(geo_loss)

transforms = v2.Compose([
    v2.AutoAugment()
])

batch_size_data = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size_data)
criterion = MSELoss()
optimizer = optim.AdamW(geo_predictor.parameters(), lr=1e-5, weight_decay=4e-6, amsgrad=True)

epochs = 20
losses = []
val_losses = []
min_val_loss = 1e8
counter = 0
def clip_grads():
    nn.utils.clip_grad_norm_(geo_predictor.parameters(), max_norm=5.0)

for epoch in track(range(epochs)):

    geo_predictor.train()
    running_loss = 0.0
    for images, coords in train_loader:
        optimizer.zero_grad()
        outputs = geo_predictor(transforms(images))
        loss = criterion(outputs, coords)
        if loss.isnan:
            loss = criterion(outputs/1000, coords)
        loss.backward()
        clip_grads()
        optimizer.step()
        running_loss += loss.item()

    losses.append(running_loss / len(train_loader))

    # Validation
    geo_predictor.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, coords in test_loader:
            outputs = geo_predictor(images)
            running_val_loss += haversine_loss(outputs, coords).cpu().numpy()

        val_losses.append(running_val_loss / len(test_loader))
    
    # Update the popout window with the latest epoch and validation loss
    update_loss_label(loss_label, epoch + 1, val_losses[-1], min_val_loss)
    
    if val_losses[-1] < min_val_loss:
        min_val_loss = val_losses[-1]
        counter = 0
    else:
        counter += 1
        
    # scheduler.step(val_losses[-1].item())
    
    if (epoch + 1) % 20 == 0 and val_losses[-1].item() < 1.1 * np.min(val_losses) and val_losses[-1].item() < 1000:
        torch.save(geo_predictor.state_dict(), f"Vojta/Temp/TMP/geo_predictor_nn_{epoch + 1}e_{batch_size_data}b_time{time.time()}.pth")
        
    evaluate_nn_model_light(geo_predictor)


print('Finished Training')
final_val_loss = val_losses[-1]
final_val_loss = int(np.round(final_val_loss, 0))
name = f'geo_predictor_nn_{epochs}e_{batch_size_data}b_{final_val_loss}k'
print(f"Saving model as {name}")
torch.save(geo_predictor.state_dict(), f"Vojta/Temp/TMP/{name}.pth")

#######################################
# Plot Training and Validation Losses   #
#######################################
plt.figure(figsize=(10, 6))
plt.plot(losses, color='skyblue')
plt.plot(val_losses, color='orange')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

#######################################
# Evaluation and Visualization          #
#######################################


def evaluate_nn_model(geo_predictor):
    print("Evaluating the neural network model...")

    geo_predictor.eval()
    y_pred = np.array([[]]).reshape(0,2)
    y_test = np.array([[]]).reshape(0,2)
    distances = np.array([]).reshape(0)
    with torch.no_grad():
        for X, y in test_loader:
            predicted_coords = geo_predictor(X).cpu().numpy()

            predicted_coords[:, 0] = (predicted_coords[:, 0] + 90) % 180 - 90
            predicted_coords[:, 1] = (predicted_coords[:, 1] + 180) % 360 - 180

            y_pred = np.append(y_pred, predicted_coords, axis=0)
            y_test = np.append(y_test, y.cpu().numpy(), axis=0)

            distances = np.append(distances, np.array([haversine_distance(y[i].cpu().numpy(), predicted_coords[i]) for i in range(y.shape[0])]))
    
    print(f"Mean Distance Error: {np.mean(distances)} km")
    print(f"Median Distance Error: {np.median(distances)} km")
    print(f"Max Distance Error: {np.max(distances)} km")
    print(f"Min Distance Error: {np.min(distances)} km")
    print(f"Standard Deviation: {np.std(distances)} km")
    print(f"25th Percentile: {np.percentile(distances, 25)} km")
    print(f"50th Percentile: {np.percentile(distances, 50)} km")
    print(f"75th Percentile: {np.percentile(distances, 75)} km")
    print(f"90th Percentile: {np.percentile(distances, 90)} km")
    print(f"Index of the minimum distance: {np.argmin(distances)} at {y_test[np.argmin(distances)]} and distance {np.min(distances)} km")
    print(f"Index of the maximum distance: {np.argmax(distances)} at {y_test[np.argmax(distances)]} and distance {np.max(distances)} km")
    
    plt.figure(figsize=(20,9))
    plt.hist(distances, bins=100, color='skyblue', edgecolor='black', linewidth=1)
    plt.title('Histogram of Distance Errors (NN)')
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.show()

    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

    plt.figure(figsize=(10, 8))
    world = gpd.read_file(url)
    world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')

    plt.scatter(y_test[:100, 1], y_test[:100, 0], color='blue', label='True')
    plt.scatter(y_pred[:100, 1], y_pred[:100, 0], color='red', label='Predicted')
    for i in range(100):
        plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='green', alpha=0.5)
    plt.title('True vs Predicted Coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

    indices = np.argsort(distances)[:100]
    true_coords_smallest = y_test[indices]
    predicted_coords_smallest = y_pred[indices]



    plt.figure(figsize=(10, 8))
    world = gpd.read_file(url)
    world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')
    plt.scatter(true_coords_smallest[:, 1], true_coords_smallest[:, 0], color='blue', label='True')
    plt.scatter(predicted_coords_smallest[:, 1], predicted_coords_smallest[:, 0], color='red', label='Predicted')
    for i in range(100):
        plt.plot([true_coords_smallest[i, 1], predicted_coords_smallest[i, 1]],
                [true_coords_smallest[i, 0], predicted_coords_smallest[i, 0]],
                color='green', alpha=0.5)
    plt.title('100 Smallest Haversine Distances: True vs Predicted Coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

    return np.mean(distances), distances

modelnames = []
for file in os.listdir("Vojta/Temp/TMP/"):
    if file.startswith("geo_predictor_nn_") and file.endswith(".pth"):
        modelnames.append(file)
        
print(f"Found {len(modelnames)} models to evaluate: {modelnames}")

for modelname in modelnames:
    print(f"Evaluating model: {modelname}")
    geo_predictor.load_state_dict(torch.load(f"Vojta/Temp/TMP/{modelname}"))
    geo_predictor.eval()
    
    # Evaluate the model
    mean_haversine_distance_nn, haversine_distances = evaluate_nn_model(geo_predictor)
    print(f"Mean Haversine Distance with NN: {mean_haversine_distance_nn} km")


# Move all the models from TMP to Models 
for file in os.listdir("Vojta/Temp/TMP/"):
    if file.startswith("geo_predictor_nn_") and file.endswith(".pth"):
        os.rename(f"Vojta/Temp/TMP/{file}", f"Vojta/Temp/Models/{file}")
        print(f"Moved {file} to Models folder.")

# Optionally, when everything is done, you can close the popout window:
loss_root.quit()
