import os
import threading
import tkinter as tk
from tkinter import font
import time

import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import geopy.distance
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

#######################################
# Tkinter Popout Window for Val Loss  #
#######################################

def create_loss_window():
    root = tk.Tk()
    root.title("Latest Validation Loss")
    large_font = font.Font(family="Helvetica", size=72, weight="bold")
    loss_label = tk.Label(root,
                          text="Epoch: N/A\nVal Loss: N/A\nLowest Loss: N/A",
                          font=large_font, bg="white", fg="black")
    loss_label.pack(padx=20, pady=20)
    return root, loss_label


def update_loss_label(loss_label, epoch, new_loss, min_val_loss):
    points = np.floor(5000 * np.exp(-1 * new_loss / 2000))
    
    # if points is nan, quit
    if np.isnan(points):
        print("Points is NaN, quitting...")
        quit()
    
    loss_label.config(
        text=f"Epoch: {epoch}\nVal Loss: {new_loss:.1f} km\nLowest Loss: {min_val_loss:.1f} km\nGeoguessr Points: {int(points)}"
    )
    loss_label.update_idletasks()
    
def launch_loss_window(window_ready_event):
    global loss_root, loss_label
    loss_root, loss_label = create_loss_window()
    window_ready_event.set()  # signal that window is ready
    loss_root.mainloop()

#######################################
# Utility: Extract Coordinates        #
#######################################

def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

#######################################
# Custom Dataset                     #
#######################################

class DualResStreetviewDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform_large = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_small = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return 2 * len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx // 2]
        image = Image.open(path).convert("RGB")
        if idx % 2 == 0:
            return self.transform_large(image)
        else:
            return self.transform_small(image)

#######################################
# Custom Models                      #
#######################################

class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super(GeoEmbeddingModel, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)


class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout0 = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc3 = nn.Linear(512, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc4 = nn.Linear(256, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(0.25)
        
        self.fc5 = nn.Linear(128, 32)
        self.batch_norm5 = nn.BatchNorm1d(32)
        self.gelu5 = nn.GELU()
        self.dropout5 = nn.Dropout(0.25)
        
        self.fc6 = nn.Linear(32, 16)
        self.batch_norm6 = nn.BatchNorm1d(16)
        self.gelu6 = nn.GELU()
        self.dropout6 = nn.Dropout(0.1)
        
        self.fc7 = nn.Linear(16, 2)

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
        return x

#######################################
# Loss and Utility Functions         #
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

#######################################
# Main Execution                     #
#######################################
def main():
    # Start Tkinter window
    window_ready_event = threading.Event()
    threading.Thread(target=launch_loss_window, args=(window_ready_event,), daemon=True).start()
    window_ready_event.wait()

    # Paths and device
    location = "D:/GeoGuessrGuessr/geoguesst"
    embeddings_file = 'Roy/ML/embeddings.npy'
    image_paths_file = 'Roy/ML/image_paths.npy'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load or generate embeddings
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file).astype(np.float32)
        image_paths = np.load(image_paths_file)
    else:
        image_paths = [os.path.join(location, img_file) for img_file in os.listdir(location)]
        model = GeoEmbeddingModel().to(device)
        if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth'):
            model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth'))
        dataset = DualResStreetviewDataset(image_paths=image_paths)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        embeddings_list = []
        model.eval()
        with torch.no_grad():
            for images in dataloader:
                images = images.to(device)
                output = model(images)
                embeddings_list.append(output.cpu().numpy().astype(np.float32))
        embeddings = np.vstack(embeddings_list)
        np.save(embeddings_file, embeddings)
        np.save(image_paths_file, np.array(image_paths))
        torch.save(model.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

    # Prepare data split
    coordinates = np.array([extract_coordinates(path) for path in image_paths])
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, coordinates,
        test_size=4000/embeddings.shape[0],
        shuffle=True,
        random_state=0
    )

    X_train = torch.tensor(X_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)

    # Initialize predictor
    geo_predictor = GeoPredictorNN().to(device)
    optimizer = optim.AdamW(geo_predictor.parameters(), lr=1e-4, weight_decay=5e-5, amsgrad=True)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=8,
        factor=0.95,
        threshold=0.01,
        threshold_mode='rel',
        verbose=True
    )

    # Training loop
    batch_size_data = 444
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size_data, shuffle=True)
    epochs = 750
    val_losses = []
    min_val_loss = float('inf')

    for epoch in range(epochs):
        geo_predictor.train()
        running_loss = 0.0
        for embeddings_batch, coords_batch in train_loader:
            optimizer.zero_grad()
            outputs = geo_predictor(embeddings_batch)
            loss = haversine_loss(outputs, coords_batch)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        geo_predictor.eval()
        with torch.no_grad():
            val_loss = haversine_loss(geo_predictor(X_test), y_test)
            val_losses.append(val_loss.item())

        update_loss_label(loss_label, epoch + 1, val_loss.item(), min_val_loss)
        if val_loss.item() < min_val_loss:
            if torch.isnan(val_loss):
                quit()
            min_val_loss = val_loss.item()
        scheduler.step(val_loss.item())

        if (epoch + 1) % 20 == 0:
            torch.save(geo_predictor.state_dict(),
                       f'Roy/ML/Saved_Models/Checkpoint_Models_NN/geo_predictor_nn_{epoch}_loss_{int(val_loss.item())}.pth')

    # Final model save
    final_val = int(round(val_losses[-1], 0))
    name = f'geo_predictor_nn_{epochs}e_{batch_size_data}b_{final_val}k'
    if os.path.exists(f'Roy/ML/Saved_Models/{name}.pth'):
        name += f'_{int(time.time())}'
    torch.save(geo_predictor.state_dict(), f'Roy/ML/Saved_Models/{name}.pth')
    
    

if __name__ == "__main__":
    main()
