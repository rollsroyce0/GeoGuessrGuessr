import os, threading, warnings, time, math
import tkinter as tk
from tkinter import font

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from rich.progress import track
import matplotlib.pyplot as plt
import geopandas as gpd

warnings.filterwarnings("ignore")

def create_loss_window():
    root = tk.Tk()
    root.title("Average-Color Geo Loss")
    large_font = font.Font(family="Helvetica", size=72, weight="bold")
    label = tk.Label(root, text="Epoch: N/A\nVal Loss: N/A\nLowest: N/A",
                     font=large_font, bg="white", fg="black")
    label.pack(padx=20, pady=20)
    return root, label

def update_loss_label(label, epoch, loss, best):
    pts = np.floor(5000 * np.exp(-loss/2000))
    label.config(text=f"Epoch: {epoch}\nVal Loss: {loss:.1f} km\nLowest: {best:.1f} km\nPts: {int(pts)}")
    label.update_idletasks()

def launch_loss_window(evt):
    global loss_root, loss_label
    loss_root, loss_label = create_loss_window()
    evt.set()
    loss_root.mainloop()

# start GUI
evt = threading.Event()
threading.Thread(target=launch_loss_window, args=(evt,), daemon=True).start()
evt.wait()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Average-Color Embeddings ---
emb_file = 'Roy/ML/Avg_Color_embeddings.npy'
paths_file = 'Roy/ML/AVGCOL_image_paths.npy'
avg_colors = np.load(emb_file).astype(np.float32)
img_paths  = np.load(paths_file)

# --- Utility ---
def extract_coordinates(path):
    lat = float(path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\',''))
    lon = float(path.split('_')[1])
    return lat, lon

coords = np.array([extract_coordinates(p) for p in img_paths], dtype=np.float32)

# --- Split Data ---
X_tr, X_te, y_tr, y_te = train_test_split(
    avg_colors, coords,
    test_size=0.2, random_state=0
)

# --- DataLoader ---
tensor_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
loader = DataLoader(tensor_tr, batch_size=1024, shuffle=True)
X_test_t = torch.from_numpy(X_te).to(device)
y_test_t = torch.from_numpy(y_te).to(device)

# --- Model ---
class AvgColorPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(avg_colors.shape[1], 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),             nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

model = AvgColorPredictor().to(device)

# --- Loss & Optimizer ---
def hav_loss(p, t):
    lat1, lon1 = torch.deg2rad(t[:,0]), torch.deg2rad(t[:,1])
    lat2, lon2 = torch.deg2rad(p[:,0]), torch.deg2rad(p[:,1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    return (6371.01 * 2 * torch.asin(torch.sqrt(a))).mean()

criterion = hav_loss
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5, amsgrad=True)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.9, verbose=True)

# --- Training Loop ---
epochs = 100
best = float('inf')
for ep in track(range(1, epochs+1), description="AvgColor Training"):
    model.train()
    loss_acc = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        opt.step()
        loss_acc += loss.item() * xb.size(0)
    train_loss = loss_acc / len(loader.dataset)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_test_t), y_test_t).item()
    if val_loss < best:
        best = val_loss
        torch.save(model.state_dict(), 'Roy/ML/Saved_Models/avgcolor_best.pth')

    sched.step(val_loss)
    update_loss_label(loss_label, ep, val_loss, best)

# close GUI
loss_root.quit()

# --- Final Evaluation ---
model.load_state_dict(torch.load('Roy/ML/Saved_Models/avgcolor_best.pth'))
model.eval()
with torch.no_grad():
    out = model(X_test_t).cpu().numpy()
    truth = y_te
    # haversine error
    def hav_km(a,b):
        la1,lo1,la2,lo2 = map(math.radians, [*a, *b])
        dlat,dlon=la2-la1, lo2-lo1
        aa=math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
        return 6371.01*2*math.atan2(math.sqrt(aa), math.sqrt(1-aa))
    errs = [hav_km(t,p) for t,p in zip(truth,out)]

print(f"Avg Err: {np.mean(errs):.3f} km, Median: {np.median(errs):.3f} km")

# --- Plotting ---
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(figsize=(10,6))
world.boundary.plot(ax=ax, linewidth=1)
N = min(len(errs), 50)
ax.scatter(truth[:N,1], truth[:N,0], label='True', alpha=0.7)
ax.scatter(out[:N,1],   out[:N,0],   label='Pred', alpha=0.7)
for i in range(N): ax.plot([truth[i,1],out[i,1]], [truth[i,0],out[i,0]], 'gray', alpha=0.5)
ax.legend(); plt.show()
