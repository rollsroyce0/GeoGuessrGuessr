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
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
import geopy.distance
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.optim as optim
import geopandas as gpd
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")

# --- Original utility and model definitions unchanged ---
#######################################
# Tkinter Popout Window for Val Loss  #
#######################################

def create_loss_window():
    root = tk.Tk()
    root.title("Latest Validation Loss")
    large_font = font.Font(family="Helvetica", size=72, weight="bold")
    loss_label = tk.Label(root, text="Epoch: N/A\nVal Loss: N/A\nLowest Loss: N/A", font=large_font, bg="white", fg="black")
    loss_label.pack(padx=20, pady=20)
    return root, loss_label


def update_loss_label(loss_label, epoch, new_loss, min_val_loss):
    points = np.floor(5000 * np.exp(-1 * new_loss / 2000))
    loss_label.config(text=f"Epoch: {epoch}\nVal Loss: {new_loss:.1f} km\nLowest Loss: {min_val_loss:.1f} km\nGeoguessr Points: {int(points)}")
    loss_label.update_idletasks()


def launch_loss_window(window_ready_event):
    global loss_root, loss_label
    loss_root, loss_label = create_loss_window()
    window_ready_event.set()
    loss_root.mainloop()

#######################################
# Utility: Extract Coordinates        #
#######################################

def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

#######################################
# Dataset and Models                  #
#######################################
class DualResStreetviewDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform_large = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.transform_small = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self): return 2*len(self.image_paths)
    def __getitem__(self, idx):
        path=self.image_paths[idx//2]
        img=Image.open(path).convert("RGB")
        return self.transform_large(img) if idx%2==0 else self.transform_small(img)

class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=models.resnet152(pretrained=True)
        self.backbone=nn.Sequential(*list(self.backbone.children())[:-1])
    def forward(self,x):
        x=self.backbone(x); return x.view(x.size(0),-1)

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2048,1024); self.dropout0=nn.Dropout(0.2)
        self.batch_norm1=nn.BatchNorm1d(1024); self.gelu1=nn.GELU(); self.dropout1=nn.Dropout(0.2)
        self.fc2=nn.Linear(1024,512); self.batch_norm2=nn.BatchNorm1d(512); self.gelu2=nn.GELU(); self.dropout2=nn.Dropout(0.2)
        self.fc3=nn.Linear(512,256); self.batch_norm3=nn.BatchNorm1d(256); self.gelu3=nn.GELU(); self.dropout3=nn.Dropout(0.2)
        self.fc4=nn.Linear(256,128); self.batch_norm4=nn.BatchNorm1d(128); self.gelu4=nn.GELU(); self.dropout4=nn.Dropout(0.2)
        self.fc5=nn.Linear(128,32);  self.batch_norm5=nn.BatchNorm1d(32);  self.gelu5=nn.GELU(); self.dropout5=nn.Dropout(0.2)
        self.fc6=nn.Linear(32,16);   self.batch_norm6=nn.BatchNorm1d(16);   self.gelu6=nn.GELU(); self.dropout6=nn.Dropout(0.1)
        self.fc7=nn.Linear(16,2)
    def forward(self,x):
        x=self.dropout0(self.gelu1(self.batch_norm1(self.fc1(x))))
        x=self.dropout2(self.gelu2(self.batch_norm2(self.fc2(x))))
        x=self.dropout3(self.gelu3(self.batch_norm3(self.fc3(x))))
        x=self.dropout4(self.gelu4(self.batch_norm4(self.fc4(x))))
        x=self.dropout5(self.gelu5(self.batch_norm5(self.fc5(x))))
        x=self.dropout6(self.gelu6(self.batch_norm6(self.fc6(x))))
        return self.fc7(x)

# Haversine loss

def degrees_to_radians(deg): return deg*0.017453292519943295

def haversine_loss(coords1,coords2):
    lat1,lon1,lat2,lon2=map(degrees_to_radians,[coords1[:,0],coords1[:,1],coords2[:,0],coords2[:,1]])
    dlat=lat2-lat1; dlon=lon2-lon1
    a=torch.sin(dlat/2)**2+torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    c=2*torch.arcsin(torch.sqrt(a)); return (6371.01*c).mean()

# Start window thread
window_ready_event=threading.Event()
threading.Thread(target=launch_loss_window,args=(window_ready_event,),daemon=True).start()
window_ready_event.wait()

# Device & paths
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
location="D:/GeoGuessrGuessr/geoguesst"
emb_file='Roy/ML/embeddings.npy'; paths_file='Roy/ML/image_paths.npy'

# Load or gen embeddings
if os.path.exists(emb_file):
    embeddings=np.load(emb_file).astype(np.float32)
    image_paths=np.load(paths_file)
else:
    image_paths=[os.path.join(location,f) for f in os.listdir(location)]
    model=GeoEmbeddingModel().to(device); dataset=DualResStreetviewDataset(image_paths); loader=DataLoader(dataset,batch_size=16,shuffle=True)
    if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth'):
        model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth'))
    embeddings_list=[]; model.eval()
    with torch.no_grad():
        for imgs in track(loader,description="Processing..."):
            embeddings_list.append(model(imgs.to(device)).cpu().numpy().astype(np.float32))
    embeddings=np.vstack(embeddings_list)
    np.save(emb_file,embeddings); np.save(paths_file,np.array(image_paths))
    torch.save(model.state_dict(),'Roy/ML/Saved_Models/geo_embedding_model.pth')

# Data split
coordinates=np.array([extract_coordinates(p) for p in image_paths])
X_train,X_val,y_train,y_val=train_test_split(embeddings,coordinates,test_size=6000/embeddings.shape[0],shuffle=True,random_state=0)
X_full=np.vstack([X_train,X_val]); y_full=np.vstack([y_train,y_val])

# PredefinedSplit
test_fold=np.array([-1]*len(X_train)+[0]*len(X_val))
ps=PredefinedSplit(test_fold)

# Sklearn wrapper keeping original class and loss
class SklearnGeoCV(BaseEstimator, RegressorMixin):
    def __init__(self,weight_decay=5e-5,amsgrad=True, lr=1e-4):
        self.weight_decay=weight_decay; self.amsgrad=amsgrad
        self.lr=lr; self.epochs=200; self.batch_size=256
        self.device=device; self.best_val_loss_=None
    def fit(self,X,y):
        n_train=len(X_train)
        Xtr=torch.tensor(X[:n_train]).float(); ytr=torch.tensor(y[:n_train]).float()
        loader=DataLoader(TensorDataset(Xtr,ytr),batch_size=self.batch_size,shuffle=True)
        model=GeoPredictorNN().to(self.device); opt=optim.AdamW(model.parameters(),lr=self.lr,weight_decay=self.weight_decay,amsgrad=self.amsgrad)
        best=np.inf
        for _ in range(self.epochs):
            model.train()
            for xb,yb in loader:
                xb,yb=xb.to(self.device),yb.to(self.device); opt.zero_grad(); out=model(xb)
                loss=haversine_loss(out,yb); loss.backward(); opt.step()
            model.eval();
            with torch.no_grad():
                Xv=torch.tensor(X[n_train:]).float().to(self.device)
                yv=torch.tensor(y[n_train:]).float().to(self.device)
                val_loss=haversine_loss(model(Xv),yv).item()
            if val_loss<best: best=val_loss
        self.best_val_loss_=best;return self
    def score(self,X,y): return -self.best_val_loss_

# Run grid search
param_grid={'weight_decay':[1e-5,5e-5,1e-4],'amsgrad':[False,True], 'lr':[1e-4,1e-3, 1e-2, 1e-5]}
gs=GridSearchCV(SklearnGeoCV(),param_grid,cv=ps,scoring='neg_mean_squared_error',verbose=3,n_jobs=4)
gs.fit(X_full,y_full)
print("Best params:",gs.best_params_)
print("Best neg MSE:",gs.best_score_)

# After GS, use best params for original training loop
best_wd, best_am, best_lr=gs.best_params_['weight_decay'],gs.best_params_['amsgrad'],gs.best_params_['lr']
geo_predictor=GeoPredictorNN().to(device)
optimizer=optim.AdamW(geo_predictor.parameters(),lr=best_lr,weight_decay=best_wd,amsgrad=best_am)
scheduler=ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=0.95,threshold=0.01,threshold_mode='rel',verbose=True)

# Original training loop follows... (unchanged)
for epoch in track(range(3000),description="Training..."):
    geo_predictor.train(); run_loss=0.0
    for xb,yb in DataLoader(list(zip(torch.tensor(X_train).float(),torch.tensor(y_train).float())),batch_size=256,shuffle=True):
        xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(); out=geo_predictor(xb)
        loss=haversine_loss(out,yb); loss.backward(); optimizer.step(); run_loss+=loss.item()
    geo_predictor.eval()
    with torch.no_grad(): val_loss=haversine_loss(geo_predictor(torch.tensor(X_val).float().to(device)),torch.tensor(y_val).float().to(device))
    update_loss_label(loss_label,epoch+1,val_loss.item(),val_loss.item())
    scheduler.step(val_loss.item())
# end of file