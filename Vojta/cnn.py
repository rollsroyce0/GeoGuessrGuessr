import os
import threading
import tkinter as tk
from tkinter import font
import time

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

        # self.image_paths = [img_file for img_file in os.listdir(self.location)]
        # self.coordinates = torch.stack([self.extract_coordinates(path) for path in self.image_paths])

        # self.transform = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #     #                      std=[0.229, 0.224, 0.225])
        # ])

        # self.images = torch.stack([self.transform(decode_image(self.location + self.image_paths[idx]).float()) for idx in track(range(len(self.coordinates)), description="Loading images... ")])
        # torch.save(self.images, self.location + "/../" + "all_images_256x256.pt") # oh boy
        # torch.save(self.coordinates, self.location + "/../" + "all_coords_256x256.pt")

        self.images = torch.load(self.location + "/../" + "all_images_256x256.pt").pin_memory()
        self.coordinates = torch.load(self.location + "/../" + "all_coords_256x256.pt")

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
        self.c1 = torch.nn.Conv2d(3, 12, 9, 2)
        self.c2 = torch.nn.Conv2d(12, 48, 7)
        self.c3 = torch.nn.Conv2d(48, 96, 5)

        self.b1 = torch.nn.BatchNorm2d(3)
        self.b2 = torch.nn.BatchNorm2d(12)
        self.b3 = torch.nn.BatchNorm2d(48)

        self.d = torch.nn.Dropout(0.3)
        self.d2 = torch.nn.Dropout2d(0.3)

        self.pool = torch.nn.MaxPool2d(2)
        
        self.r = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

        self.f1 = torch.nn.Linear(37632, 256)
        self.f2 = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = self.b1(x)

        x = self.pool(self.c1(x))
        x = self.b2(x)
        x = self.d2(x)
        x = self.pool(self.c2(x))
        x = self.b3(x)
        x = self.d2(x)
        # x = self.r(self.pool(self.c3(x)))
        # x = self.r(self.pool(self.c4(x)))

        x = self.f1(x.flatten(start_dim=1))
        x = self.d(x)
        x = self.f2(self.r(x))
        
        return x

#######################################
# Define Haversine Loss and Optimizer   #
#######################################
def degrees_to_radians(deg):
    return deg * 0.017453292519943295

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


# Initialize the predictor model
geo_predictor = GeoPredictorNN().to(device)
summary(geo_predictor, (3,256,256))
geo_predictor.compile()

location = "/home/TheSmilingTurtle/Downloads/Roy_Files/Pictures/"
dataset = StreetviewDataset(location=location)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

haversine_loss = torch.compile(haversine_loss)
geo_loss = torch.compile(geo_loss)

transforms = v2.Compose([
    v2.AutoAugment()
])

batch_size_data = 2000
train_loader = DataLoader(train_dataset, batch_size=batch_size_data, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_data, shuffle=True)
criterion = MSELoss()
optimizer = optim.AdamW(geo_predictor.parameters(), lr=1, weight_decay=1e-2, amsgrad=True)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=8,
    factor=0.98,
    threshold=0.01,
    threshold_mode='rel'
)
epochs = 10
losses = []
val_losses = []
min_val_loss = 1e8
counter = 0


for epoch in range(epochs):

    geo_predictor.train()
    running_loss = 0.0
    for images, coords in track(train_loader, description="Training..."):
        optimizer.zero_grad()
        outputs = geo_predictor(transforms(images))
        loss = criterion(outputs, coords)
        if loss.isnan:
            loss = criterion(outputs/1000, coords)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    losses.append(running_loss / len(train_loader))

    # Validation
    geo_predictor.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, coords in track(test_loader, description="Testing..."):
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
        torch.save(geo_predictor.state_dict(), f'./Saved_Models/TMP/geo_predictor_nn_{epoch}_loss_{np.round(val_losses[-1].item(), 0)}.pth')


print('Finished Training')
final_val_loss = val_losses[-1]
final_val_loss = int(np.round(final_val_loss, 0))
name = f'geo_predictor_nn_{epochs}e_{batch_size_data}b_{final_val_loss}k'
print(f"Saving model as {name}")
torch.save(geo_predictor.state_dict(), f'./Saved_Models/TMP/{name}.pth')

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
def haversine_distance(coords1, coords2):
    return geopy.distance.geodesic(coords1, coords2).kilometers

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

    for X, y in test_loader:
        plt.title(str(list(y[0].cpu().numpy())))
        plt.imshow(X[0].cpu().numpy().transpose((1,2,0))/255)
        plt.show()
        plt.title(str(list(y[1].cpu().numpy())))
        plt.imshow(X[1].cpu().numpy().transpose((1,2,0))/255)
        plt.show()
        plt.title(str(list(y[2].cpu().numpy())))
        plt.imshow(X[2].cpu().numpy().transpose((1,2,0))/255)

        break
    plt.show()

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


geo_predictor.load_state_dict(torch.load(f'./Saved_Models/TMP/{name}.pth'))
mean_haversine_distance_nn, haversine_distances = evaluate_nn_model(geo_predictor)
print(f"Mean Haversine Distance with NN: {mean_haversine_distance_nn} km")

# Optionally, when everything is done, you can close the popout window:
loss_root.quit()
