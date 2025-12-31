import os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.progress import track
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# config
# -------------------------------------------------
EFILE = "Roy/ML/Resnet_Experiment/Best_embeddings.npy"
PFILE = "Roy/ML/Resnet_Experiment/Best_image_paths.npy"

BATCH = 512
EPOCHS = 100

# -------------------------------------------------
# utils
# -------------------------------------------------
def coords(p):
    a, b = os.path.basename(p).split("_")[:2]
    return float(a), float(b)

def latlon_to_ecef(y):
    lat = torch.deg2rad(y[:,0])
    lon = torch.deg2rad(y[:,1])
    cl = torch.cos(lat)
    return torch.stack([
        cl * torch.cos(lon),
        cl * torch.sin(lon),
        torch.sin(lat)
    ], dim=1)

def angular_loss(p, t):
    p = nn.functional.normalize(p, dim=1)
    t = nn.functional.normalize(t, dim=1)
    return torch.rad2deg(
        torch.acos((p * t).sum(1).clamp(-1 + 1e-7, 1 - 1e-7))
    ).mean()
    
def predict(z):
    with torch.no_grad():
        p = net(torch.tensor(z).float().unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        return [(p[0]+90)%180-90,(p[1]+180)%360-180]

# -------------------------------------------------
# dataset (embeddings only)
# -------------------------------------------------
class EmbeddingDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# -------------------------------------------------
# Mixture-of-Experts head
# -------------------------------------------------
class GeoMoE(nn.Module):
    def __init__(self, D, E=6, k=2, expert_dropout=0.1):
        """
        D: embedding dimension
        E: number of experts
        k: top-k experts to route to
        expert_dropout: probability to drop an expert during training
        """
        super().__init__()
        self.E = E
        self.k = k
        self.expert_dropout = expert_dropout

        # ---- router (deep + wide) ----
        self.router = nn.Sequential(
            nn.Linear(D, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, E)
        )

        # ---- experts (deep, multiple layers) ----
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 3)
            ) for _ in range(E)
        ])

    def forward(self, x):
        # routing logits
        logits = self.router(x)  # [B, E]

        if self.training and self.expert_dropout > 0:
            # randomly drop some experts (regularization)
            drop_mask = torch.rand_like(logits) > self.expert_dropout
            logits = logits.masked_fill(~drop_mask, float('-inf'))

        # top-k routing
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=1)  # [B, k]
        w = torch.softmax(topk_vals, dim=1)  # [B, k]

        # gather expert outputs
        outs = torch.stack([self.experts[i](x) for i in range(self.E)], dim=1)  # [B, E, 3]
        selected = torch.gather(
            outs, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, k, 3]

        # weighted sum
        out = (w.unsqueeze(-1) * selected).sum(dim=1)

        # residual connection: mean of all experts
        out += outs.mean(dim=1)

        return out



# -------------------------------------------------
# load embeddings
# -------------------------------------------------
print("Loading embeddings...")
X = np.load(EFILE).astype(np.float32)
paths = np.load(PFILE, allow_pickle=True)

Y = np.array([coords(p) for p in paths], dtype=np.float32)
Y = latlon_to_ecef(torch.tensor(Y)).numpy()

print("Embeddings:", X.shape)

# -------------------------------------------------
# split
# -------------------------------------------------
Xtr, Xte, Ytr, Yte = train_test_split(
    X, Y, test_size=2000/len(X), random_state=0
)

train_ds = EmbeddingDS(Xtr, Ytr)
test_ds  = EmbeddingDS(Xte, Yte)

train_dl = DataLoader(
    train_ds, BATCH, shuffle=True,
    num_workers=0, pin_memory=True
)

test_dl = DataLoader(
    test_ds, BATCH, shuffle=False,
    num_workers=0, pin_memory=True
)

# -------------------------------------------------
# model / optimizer
# -------------------------------------------------
EMB_DIM = X.shape[1]
net = GeoMoE(EMB_DIM, E=6).to(DEVICE)

opt = optim.AdamW(
    net.parameters(),
    lr=2e-3,
    weight_decay=1e-5,
    amsgrad=True
)

sch = ReduceLROnPlateau(opt, "min", factor=0.5, patience=5, min_lr=1e-7, verbose=True)
scaler = torch.cuda.amp.GradScaler()

# -------------------------------------------------
# training
# -------------------------------------------------
best = 1e9

for e in track(range(EPOCHS), "Training"):
    net.train()

    for x, y in train_dl:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            pred = net(x)
            loss = angular_loss(pred, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    # ---------- validation ----------
    net.eval()
    with torch.no_grad():
        vals = []
        for x, y in test_dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            vals.append(angular_loss(net(x), y).item())
        vl = float(np.mean(vals))
        
    # plot 10 example point on a world map
    if (e+1) % 10 == 0:
        gdf_real = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
            [coords(paths[i])[1] for i in range(len(paths)) if i % (len(paths)//100) == 0],
            [coords(paths[i])[0] for i in range(len(paths)) if i % (len(paths)//100) == 0],
            crs="EPSG:4326"
            
        ))
        gdf_pred = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
            [predict( torch.tensor(X[i]).float() )[1] for i in range(len(paths)) if i % (len(paths)//100) == 0],
            [predict( torch.tensor(X[i]).float() )[0] for i in range(len(paths)) if i % (len(paths)//100) == 0],
            crs="EPSG:4326"
        ))
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        ax = world.plot(figsize=(12, 8), color='white', edgecolor='black')
        gdf_real.sample(10).plot(ax=ax, color='blue', markersize=50, label='Real')
        gdf_pred.sample(10).plot(ax=ax, color='red', markersize=50, label='Predicted')
        plt.legend()    
        plt.title(f'Epoch {e+1}: Real vs Predicted Locations')
        plt.show(block =False)
        plt.pause(2)
        plt.close()

    sch.step(vl)

    if vl < best:
        best = vl
    torch.save(net.state_dict(), f"Roy/ML/Resnet_Experiment/Saved_Models_New/Checkpoint_Models_NN/geoMoE_check_{e+1:03d}e_{BATCH}b_{best:.2f}_{time.time()}.pth")

    print(f"Epoch {e:03d} | mean angular error: {vl:.2f}°")
import geopy.distance

def ecef_to_latlon_tensor(v):
    x, y, z = v[:,0], v[:,1], v[:,2]
    lat = torch.atan2(z, torch.sqrt(x*x + y*y))
    lon = torch.atan2(y, x)
    return torch.rad2deg(lat), torch.rad2deg(lon)

#save final model
torch.save(net.state_dict(), f"Roy/ML/Resnet_Experiment/Saved_Models_New/geoMoE_{e+1:03d}e_{BATCH}b_{best:.2f}_{time.time()}.pth")
# compute mean geodesic distance in km
distances = []
net.eval()
with torch.no_grad():
    for x, y in test_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred_ecef = net(x)
        lat_pred, lon_pred = ecef_to_latlon_tensor(pred_ecef)
        lat_true, lon_true = ecef_to_latlon_tensor(y)
        for i in range(len(lat_pred)):
            distances.append(
                geopy.distance.geodesic(
                    (lat_true[i].item(), lon_true[i].item()),
                    (lat_pred[i].item(), lon_pred[i].item())
                ).km
            )
mean_km = np.mean(distances)
print("Mean geodesic error: {:.2f} km".format(mean_km))

