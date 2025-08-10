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

def create_loss_window():
    root = tk.Tk()
    root.title("Latest Validation Loss")
    large_font = font.Font(family="Helvetica", size=72, weight="bold")
    loss_label = tk.Label(root, text="Epoch: N/A\nVal Loss: N/A\nLowest Loss: N/A", font=large_font, bg="white", fg="black")
    loss_label.pack(padx=20, pady=20)
    return root, loss_label

def update_loss_label(loss_label, epoch, new_loss, min_val_loss):
    points =np.floor(5000 * np.exp(-1*new_loss/2000))
    loss_label.config(text=f"Epoch: {epoch}\nVal Loss: {new_loss:.1f} km\nLowest Loss: {min_val_loss:.1f} km\nGeoguessr Points: {int(points)}")
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


#######################################
# Utility: Extract Coordinates        #
#######################################
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

#######################################
# Check Device and Set Location         #
#######################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")

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

#######################################
# Custom Dataset: DualResStreetviewDataset
#######################################
class LargeResDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

class SmallResDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

class GeoEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super(GeoEmbeddingModel, self).__init__()
        backbone = models.resnet152(pretrained=True)
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x

embeddings_file = 'Roy/ML/Playground/Double/embeddings.npy'
image_paths_file = 'Roy/ML/Playground/Double/image_paths.npy'

if os.path.exists(embeddings_file) and os.path.exists(image_paths_file):
    embeddings = np.load(embeddings_file).astype(np.float32)
    image_paths = np.load(image_paths_file, allow_pickle=True)
    coordinates = np.array([extract_coordinates(path) for path in image_paths])
else:
    image_paths = [os.path.join(location, img_file) for img_file in os.listdir(location)]
    coordinates = np.array([extract_coordinates(path) for path in image_paths])
    model = GeoEmbeddingModel().to(device)
    
    large_loader = DataLoader(LargeResDataset(image_paths), batch_size=8, shuffle=True)
    small_loader = DataLoader(SmallResDataset(image_paths), batch_size=8, shuffle=True)

    
    if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model.pth'):
        model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model.pth'))

    print("Generating custom embeddings...")
    embeddings_list = []
    model.eval()
    with torch.no_grad():
        for images in track(zip(large_loader, small_loader), description="Generating embeddings..."):
            large_images, small_images = images
            large_images = large_images.to(device)
            small_images = small_images.to(device)
            
            large_embeddings = model(large_images)
            small_embeddings = model(small_images)
            
            combined_embeddings = torch.cat((large_embeddings, small_embeddings), dim=1)
            embeddings_list.append(combined_embeddings.cpu().numpy())
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    np.save(embeddings_file, embeddings)
    np.save(image_paths_file, np.array(image_paths), allow_pickle=True)
    torch.save(model.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

#print(f"Number of images: {len(image_paths)}")
#print(f"Embeddings shape: {embeddings.shape}")

#######################################
# Prepare Coordinates and Data Split    #
#######################################
coordinates = np.array([extract_coordinates(path) for path in image_paths])
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, coordinates,
    test_size=2000/embeddings.shape[0],
    shuffle=True,
    random_state=0
)

X_train = torch.tensor(X_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)

#######################################
# Define the GeoPredictorNN Model       #
#######################################
class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.dropout0 = nn.Dropout(0.05)

        self.batch_norm1 = nn.BatchNorm1d(2048)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.15)

        self.fc2 = nn.Linear(2048, 1024)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(1024, 512)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(512, 256)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(256, 128)
        self.batch_norm5 = nn.BatchNorm1d(128)
        self.gelu5 = nn.GELU()
        self.dropout5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(128, 32)
        self.batch_norm6 = nn.BatchNorm1d(32)
        self.gelu6 = nn.GELU()
        self.dropout6 = nn.Dropout(0.2)

        self.fc7 = nn.Linear(32, 16)
        self.batch_norm7 = nn.BatchNorm1d(16)
        self.gelu7 = nn.GELU()
        self.dropout7 = nn.Dropout(0.05)

        self.fc8 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout0(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.gelu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.batch_norm4(x)
        x = self.gelu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.batch_norm5(x)
        x = self.gelu5(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        x = self.batch_norm6(x)
        x = self.gelu6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.batch_norm7(x)
        x = self.gelu7(x)
        x = self.dropout7(x)

        x = self.fc8(x)
        return x

# Initialize the predictor model
geo_predictor = GeoPredictorNN().to(device)
# print a summary of the model
#from torchinfo import summary
#summary(geo_predictor, input_size=(1, 2048), device=device.type)
#from torchinfo import summary
#summary(geo_predictor, input_size=(1, 2048), device=device.type)

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
    distance = distance
    return distance.mean()

criterion = haversine_loss
optimizer = optim.AdamW(geo_predictor.parameters(), lr=1e-4, weight_decay=8e-5, amsgrad=True)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.95,
    threshold=0.01,
    threshold_mode='rel', 
    verbose=True
)

#######################################
# Training Loop                         #
#######################################

batch_size_data = 256
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size_data, shuffle=True)
epochs = 567

losses = []
val_losses = []
min_val_loss = 1e5
counter = 0

for epoch in track(range(epochs), description="Training the model..."):

    geo_predictor.train()
    running_loss = 0.0
    for embeddings_batch, coords_batch in train_loader:
        embeddings_batch = embeddings_batch.float().to(device)
        coords_batch = coords_batch.float().to(device)
        optimizer.zero_grad()
        outputs = geo_predictor(embeddings_batch)
        loss = criterion(outputs, coords_batch)
        if torch.isnan(loss):
            print(f"Loss is NaN, skipping this batch... at epoch {epoch}")
            
            break

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss / len(train_loader))
    
    if torch.isnan(torch.tensor(losses[-1])):
        print(f"Loss is NaN, skipping this epoch... at epoch {epoch}")
        losses[-1] = 1e5  # Append a large value to avoid saving
        continue
    
    # Validation
    geo_predictor.eval()
    with torch.no_grad():
        val_loss = haversine_loss(geo_predictor(X_test), y_test)
        val_losses.append(val_loss.item())
    
    # Update the popout window with the latest epoch and validation loss
    update_loss_label(loss_label, epoch + 1, val_loss.item(), min_val_loss)
    
    if val_loss.item() < min_val_loss:
        min_val_loss = val_loss.item()
        counter = 0
    else:
        counter += 1
        
    scheduler.step(val_loss.item())
    
    if epoch % 25 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {losses[-1]:.4f}, Val Loss: {val_loss.item():.4f}, Min Val Loss: {min_val_loss:.4f}, Counter: {counter}")
    
    geo_predictor.eval()
    if torch.isnan(val_loss):
        print(f"Validation loss is NaN, skipping saving model... at epoch {epoch}")
        val_losses.append(1e5)  # Append a large value to avoid saving
        losses.append(1e5)  # Append a large value to avoid saving
        continue
    if val_loss.item() < 1500 and epoch % 1 == 0:
        name = f'geo_predictor_nn_{epoch}e_{batch_size_data}b_{int(np.round(val_loss.item(), 0))}k_checkpoint_{int(time.time())}'
        torch.save(geo_predictor.state_dict(), f'Roy/ML/Saved_Models_New/Checkpoint_Models_NN/{name}.pth')

#print('Finished Training')
final_val_loss = val_losses[-1]
final_val_loss = int(np.round(final_val_loss, 0))
name = f'geo_predictor_nn_{epochs}e_{batch_size_data}b_{final_val_loss}k'
#check if the model already exists
if os.path.exists(f'Roy/ML/Saved_Models/{name}.pth'):
    name = f'geo_predictor_nn_{epochs}e_{batch_size_data}b_{final_val_loss}k_{int(time.time())}'
    print(f"Model {name} already exists, saving with a timestamp.")

print(f"Saving model as {name}")
torch.save(geo_predictor.state_dict(), f'Roy/ML/Saved_Models/{name}.pth')


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
plt.show(block=False)
plt.pause(2)
plt.close()

#######################################
# Evaluation and Visualization          #
#######################################
def haversine_distance(coords1, coords2):
    return geopy.distance.geodesic(coords1, coords2).kilometers

def predict_coordinates_nn(embedding, geo_predictor):
    geo_predictor.eval()
    with torch.no_grad():
        embedding = torch.tensor(embedding).float().unsqueeze(0).to(device)
        predicted_coords = geo_predictor(embedding).cpu().numpy()[0]
        predicted_coords[0] = (predicted_coords[0] + 90) % 180 - 90
        predicted_coords[1] = (predicted_coords[1] + 180) % 360 - 180
    return predicted_coords

def evaluate_nn_model(X_test, y_test, geo_predictor):
    print("Evaluating the neural network model...")
    y_pred = np.array([predict_coordinates_nn(embedding, geo_predictor) for embedding in track(X_test, description="Predicting coordinates...")])
    distances = np.array([haversine_distance(y_test[i], y_pred[i]) for i in range(len(y_test))])
    print(f"Mean Distance Error: {np.mean(distances)} km")
    print(f"Median Distance Error: {np.median(distances)} km")
    print(f"Max Distance Error: {np.max(distances)} km")
    print(f"Min Distance Error: {np.min(distances)} km")
    print(f"Standard Deviation: {np.std(distances)} km")
    print(f"25th Percentile: {np.percentile(distances, 25)} km")
    print(f"50th Percentile: {np.percentile(distances, 50)} km")
    print(f"75th Percentile: {np.percentile(distances, 75)} km")
    print(f"90th Percentile: {np.percentile(distances, 90)} km")
    print(f"Index of the minimum distance: {np.argmin(distances)} with name {image_paths[np.argmin(distances)]} and distance {np.min(distances)} km")
    print(f"Index of the maximum distance: {np.argmax(distances)} with name {image_paths[np.argmax(distances)]} and distance {np.max(distances)} km")
    
    plt.figure(figsize=(20,9))
    plt.hist(distances, bins=100, color='skyblue', edgecolor='black', linewidth=1)
    plt.title('Histogram of Distance Errors (NN)')
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    return np.mean(distances), distances

geo_predictor.load_state_dict(torch.load(f'Roy/ML/Saved_Models/{name}.pth'))
mean_haversine_distance_nn, haversine_distances = evaluate_nn_model(X_test, y_test, geo_predictor)
print(f"Mean Haversine Distance with NN: {mean_haversine_distance_nn} km")


X_test = X_test.cpu().numpy()
y_test = y_test.cpu().numpy()


plt.figure(figsize=(10, 8))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')
plt.scatter(y_test[:100, 1], y_test[:100, 0], color='blue', label='True')
y_pred = np.array([predict_coordinates_nn(embedding, geo_predictor) for embedding in X_test[:100]])
plt.scatter(y_pred[:100, 1], y_pred[:100, 0], color='red', label='Predicted')
for i in range(100):
    plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='green', alpha=0.5)
plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show(block=False)
plt.pause(2)
plt.close()

indices = np.argsort(haversine_distances)[:100]
image_paths_smallest = [image_paths[i] for i in indices]
true_coords_smallest = y_test[indices]
predicted_coords_smallest = np.array([predict_coordinates_nn(embedding, geo_predictor) for embedding in X_test[indices]])

plt.figure(figsize=(10, 8))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
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
plt.show(block=False)
plt.pause(2)
plt.close()

# Optionally, when everything is done, you can close the popout window:
loss_root.quit()
# Note: The popout window will remain open until you close it manually or the script ends. 






