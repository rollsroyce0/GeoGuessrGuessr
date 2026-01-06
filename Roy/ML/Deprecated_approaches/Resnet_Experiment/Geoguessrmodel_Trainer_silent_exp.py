import os, time, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rich.progress import track
from torch.optim.lr_scheduler import ReduceLROnPlateau
import geopy.distance


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "D:/GeoGuessrGuessr/geoguesst"

# ---------- utils ----------
def coords(p):
    a, b = p.replace(ROOT+"\\","").split("_")[:2]
    return float(a), float(b)

rad = lambda x: x * 0.017453292519943295

def haversine(p, t):
    p, t = rad(p), rad(t)
    d = p - t
    a = torch.sin(d[:,0]/2)**2 + torch.cos(p[:,0])*torch.cos(t[:,0])*torch.sin(d[:,1]/2)**2
    return (6371.01 * 2 * torch.arcsin(torch.sqrt(a))).mean()

# ---------- dataset ----------
T = lambda *x: transforms.Compose([
    transforms.Resize((1024,1024)), *x,
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

class StreetViewDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.t = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.RandomAffine(15,(.1,.1),(.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        ])

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.t(img), torch.tensor(coords(self.paths[i]), dtype=torch.float32)


# ---------- embedding ----------
class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone(x)
        return self.pool(x).flatten(1)


# ---------- predictor ----------
class GeoNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(.2),
            nn.Linear(512,128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(.2),
            nn.Linear(128,2)
        )
    def forward(self,x): 
        return self.net(x)


# ---------- load / build embeddings ----------
EFILE, PFILE = "Roy/ML/Tiny/Best_embeddings.npy", "Roy/ML/Tiny/Best_image_paths.npy"

paths = np.array([os.path.join(ROOT,f) for f in os.listdir(ROOT)])

train_p, test_p = train_test_split(paths, test_size=2000/len(paths), random_state=0)

train_dl = DataLoader(
    StreetViewDS(train_p),
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=0
)

test_ds = StreetViewDS(test_p)

# Generate embeddings for all paths
embed = Embed().to(device).eval()
X = []
with torch.no_grad():
    for imgs, _ in DataLoader(StreetViewDS(paths), batch_size=16, pin_memory=True, num_workers=0):
        X.append(embed(imgs.to(device)).cpu().numpy())
X = np.vstack(X)

Y = np.array([coords(p) for p in paths])

assert len(X) == len(Y)

Xtr,Xte,Ytr,Yte = train_test_split(
    X, Y, test_size=2000/len(X), random_state=0
)

Xtr,Xte,Ytr,Yte = map(
    lambda x: torch.tensor(x).float().to(device),
    (Xtr,Xte,Ytr,Yte)
)


# ---------- training ----------
# embed already created above
net = GeoNN().to(device)

opt = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=8e-5, amsgrad=True)
sch = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5, min_lr=1e-7)

best = 1e9

for e in track(range(500), "Training"):
    net.train()
    for imgs,y in train_dl:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            x = embed(imgs)

        opt.zero_grad()
        loss = haversine(net(x), y)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    # validation
    net.eval()
    with torch.no_grad():
        d = []
        for i in range(0,len(test_ds),16):
            imgs,y = zip(*[test_ds[j] for j in range(i,min(i+16,len(test_ds)))])
            imgs = torch.stack(imgs).to(device)
            y = torch.stack(y).to(device)
            d.append(haversine(net(embed(imgs)),y).item())
        vl = np.mean(d)

    sch.step(vl)

    if vl < best:
        best = vl
        torch.save(net.state_dict(), "geoNN_best.pth")

#save final model
torch.save(net.state_dict(), f"Roy/ML/Tiny/Saved_Models_New/geoNN_final_{e+1}e_{int(vl)}k.pth".format(e, int(vl)))

print("Best val haversine:", best, "km")
# ---------- evaluation ----------
@torch.no_grad()
def predict(img):
    x = embed(img.unsqueeze(0).to(device))
    p = net(x).cpu().numpy()[0]
    return [(p[0]+90)%180-90,(p[1]+180)%360-180]


def dist(a,b):
    return 1e4 if np.isnan(a).any() else geopy.distance.geodesic(a,b).km

Xn,Yn = Xte.cpu().numpy(), Yte.cpu().numpy()
D = np.array([dist(Yn[i],predict(Xn[i])) for i in range(len(Yn))])
print("Mean Haversine:", D.mean(),"km")
