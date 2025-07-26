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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import geopy.distance
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import geopandas as gpd
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")


# --------------------------------------
# 2. Utility: Extract Coordinates      
# --------------------------------------
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

# --------------------------------------
# 3. Device and Data Location          
# --------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
location = "D:/GeoGuessrGuessr/geoguesst"

# --------------------------------------
# 4. Transforms and Dataset            
# --------------------------------------
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class DualResStreetviewDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform_large = transform
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
        return self.transform_large(image) if idx % 2 == 0 else self.transform_small(image)

# --------------------------------------
# 5. Embedding Model                   
# --------------------------------------
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

# Load or compute embeddings
emb_file = 'Roy/ML/embeddings.npy'
paths_file = 'Roy/ML/image_paths.npy'
if os.path.exists(emb_file):
    embeddings = np.load(emb_file).astype(np.float32)
    image_paths = np.load(paths_file)
else:
    image_paths = [os.path.join(location, f) for f in os.listdir(location)]
    model_e = GeoEmbeddingModel().to(device)
    if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth'):
        model_e.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth'))
    ds = DualResStreetviewDataset(image_paths)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    embs = []
    model_e.eval()
    with torch.no_grad():
        for imgs in track(loader, description="Generating embeddings..."):
            embs.append(model_e(imgs.to(device)).cpu().numpy())
    embeddings = np.vstack(embs).astype(np.float32)
    np.save(emb_file, embeddings)
    np.save(paths_file, image_paths)
    torch.save(model_e.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

print(f"# images: {len(image_paths)} | emb shape: {embeddings.shape}")
coords = np.array([extract_coordinates(p) for p in image_paths])

# --------------------------------------
# 6. Train-test split                  
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, coords,
    test_size=10000/embeddings.shape[0],
    random_state=0,
    shuffle=True
)
X_train = torch.tensor(X_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)

# --------------------------------------
# 7. Hyperparameter Search (1/10 data)  
# --------------------------------------
n_sub = X_train.shape[0] // 20
idx_sub = np.random.choice(X_train.shape[0], n_sub, replace=False)
X_sub = X_train[idx_sub].cpu().numpy()
y_sub = y_train[idx_sub].cpu().numpy()

def degrees_to_radians(deg): return deg * np.pi / 180

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout0 = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
        
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
        self.dropout5 = nn.Dropout(0.2)
        
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

def haversine_loss(coords1, coords2):
    lat1, lon1 = coords1[:,0], coords1[:,1]
    lat2, lon2 = coords2[:,0], coords2[:,1]
    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [lat1,lon1,lat2,lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    dist = 6371.01 * c
    return dist.mean()

def haversine_distance(a,b): return geopy.distance.geodesic(a,b).km

class GeoPredictorEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=256):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        Xt = torch.tensor(X).float().to(self.device)
        yt = torch.tensor(y).float().to(self.device)
        loader = DataLoader(list(zip(Xt, yt)), batch_size=self.batch_size, shuffle=True)
        model = GeoPredictorNN().to(self.device)
        opt = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            model.train()
            for emb, coords in loader:
                opt.zero_grad()
                loss = haversine_loss(model(emb), coords)
                loss.backward()
                opt.step()
        self.model_ = model
        return self

    def predict(self, X):
        self.model_.eval()
        Xt = torch.tensor(X).float().to(self.device)
        with torch.no_grad(): out = self.model_(Xt).cpu().numpy()
        return out

    def score(self, X, y):
        preds = self.predict(X)
        dists = [haversine_distance(y[i], preds[i]) for i in range(len(y))]
        return -np.mean(dists)

param_grid = {
    'lr': [1e1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    'weight_decay': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'batch_size': [128, 256, 512, 1024, 2048, 64],
}

grid = GridSearchCV(
    estimator=GeoPredictorEstimator(),
    param_grid=param_grid,
    cv=3,
    scoring=None,
    verbose=3,
    n_jobs=8
)
print("Starting hyperparameter search on 1/10 of training data...")
grid.fit(X_sub, y_sub)
print("Best params:", grid.best_params_)
print("Best mean distance (km):", -grid.best_score_)

best = grid.best_params_
# --------------------------------------
# 8. Define full Predictor & Optimizer 
# --------------------------------------


geo_predictor = GeoPredictorNN().to(device)
optimizer = optim.AdamW(geo_predictor.parameters(), lr=best['lr'], weight_decay=best['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.95)
epochs = best['epochs']  # use optimized epoch count

# Continue with your original training & eval loop...
