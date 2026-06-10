import os, time, threading, argparse, warnings
import tkinter as tk

import torch
import numpy as np
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import geopy.distance
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

#######################################
# ARGUMENTS
#######################################
parser = argparse.ArgumentParser()
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--tk", action="store_true")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=128)
args = parser.parse_args()

QUIET = args.quiet
USE_TK = args.tk
EPOCHS = args.epochs
BATCH_SIZE = args.batch

def log(msg):
    if not QUIET:
        print(msg)

#######################################
# TKINTER WINDOW
#######################################
class LossWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Training Loss")
        self.label = tk.Label(self.root, text="Starting...", font=("Arial", 14))
        self.label.pack(padx=20, pady=20)
        self.running = True
        threading.Thread(target=self.root.mainloop, daemon=True).start()

    def update(self, epoch, loss, val_loss):
        if self.running:
            self.label.config(text=f"Epoch {epoch}\nLoss: {loss:.2f}\nVal: {val_loss:.2f}")
            self.root.update_idletasks()

    def close(self):
        self.running = False
        self.root.quit()

#######################################
# UTILS
#######################################
def extract_coordinates(path):
    lat = float(path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(path.split('_')[1])
    return lat, lon

def degrees_to_radians(deg):
    return torch.deg2rad(deg)

def haversine_loss(a, b):
    lat1, lon1 = a[:,0], a[:,1]
    lat2, lon2 = b[:,0], b[:,1]

    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [lat1, lon1, lat2, lon2])

    dlat, dlon = lat2 - lat1, lon2 - lon1
    A = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    C = 2 * torch.arcsin(torch.sqrt(A))
    return (6371.01 * C).mean()

#######################################
# DEVICE
#######################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################
# DATASET
#######################################
class DualResStreetviewDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.large = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.small = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self): return 2*len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i//2]).convert("RGB")
        return self.large(img) if i%2==0 else self.small(img)

#######################################
# EMBEDDING MODEL
#######################################
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(m.children())[:-1])

    def forward(self,x):
        return self.backbone(x).view(x.size(0),-1)

#######################################
# PREDICTOR
#######################################
class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048,1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128,32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32,16), nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(16,2)
        )

    def forward(self,x):
        x = self.net(x)
        x[:,0] = torch.tanh(x[:,0]) * 90
        x[:,1] = torch.tanh(x[:,1]) * 180
        return x

#######################################
# LOAD / BUILD EMBEDDINGS
#######################################
location = "D:/GeoGuessrGuessr/geoguesst"
emb_file = 'Roy/ML/Best_embeddings.npy'
paths_file = 'Roy/ML/Best_image_paths.npy'

if os.path.exists(emb_file):
    embeddings = np.load(emb_file).astype(np.float32)
    image_paths = np.load(paths_file, allow_pickle=True)
else:
    image_paths = [os.path.join(location,f) for f in os.listdir(location)]
    model = GeoEmbeddingModel().to(device)

    dataset = DualResStreetviewDataset(image_paths)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.eval()
    embs = []
    iterator = loader if QUIET else track(loader, description="Embedding")

    with torch.no_grad():
        for batch in iterator:
            embs.append(model(batch.to(device)).cpu().numpy())

    embeddings = np.vstack(embs)
    np.save(emb_file, embeddings)
    np.save(paths_file, np.array(image_paths))

#######################################
# PREP DATA
#######################################
coords = np.array([extract_coordinates(p) for p in image_paths])

Xtr,Xte,Ytr,Yte = train_test_split(
    embeddings, coords,
    test_size=2000/len(embeddings),
    random_state=0
)

Xtr = torch.tensor(Xtr).float().to(device)
Xte = torch.tensor(Xte).float().to(device)
Ytr = torch.tensor(Ytr).float().to(device)
Yte = torch.tensor(Yte).float().to(device)

#######################################
# TRAIN
#######################################
model = GeoPredictorNN().to(device)
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=8e-5)
sched = ReduceLROnPlateau(opt, patience=5, factor=0.95)

loader = DataLoader(list(zip(Xtr,Ytr)), batch_size=BATCH_SIZE, shuffle=True)

loss_window = LossWindow() if USE_TK else None

best = 1e9
losses, vlosses = [], []

epochs_iter = range(EPOCHS) if QUIET else track(range(EPOCHS), description="Training")

for e in epochs_iter:
    model.train()
    total=0

    for xb,yb in loader:
        opt.zero_grad()
        loss = haversine_loss(model(xb), yb)
        if torch.isnan(loss): continue
        loss.backward()
        opt.step()
        total += loss.item()

    train_loss = total/len(loader)
    losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val = haversine_loss(model(Xte), Yte).item()

    vlosses.append(val)
    sched.step(val)

    if loss_window:
        loss_window.update(e+1, train_loss, val)

    if val < best:
        best = val
        torch.save(model.state_dict(), "best_model.pth")

    if e%25==0:
        log(f"Epoch {e} | Loss {train_loss:.1f} | Val {val:.1f}")

if loss_window:
    loss_window.close()

#######################################
# EVAL
#######################################
def haversine(a,b):
    if np.isnan(a).any() or np.isnan(b).any(): return 1e4
    return geopy.distance.geodesic(a,b).km

model.load_state_dict(torch.load("best_model.pth"))

pred = model(Xte).cpu().detach().numpy()
true = Yte.cpu().numpy()

dists = np.array([haversine(true[i],pred[i]) for i in range(len(true))])
log(f"Mean distance: {dists.mean():.1f} km")