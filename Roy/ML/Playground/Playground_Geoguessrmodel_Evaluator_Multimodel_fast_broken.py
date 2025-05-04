import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# ----------------- Config -----------------
TEST_SETS = {
    'Game':       [[59.2642, 10.4276], [1.4855, 103.8676], [54.9927, -1.6732], [19.4745, -99.1974], [58.6133, 49.6275]],
    'Validation': [[43.3219, -5.5784],   [23.0376, 72.5819],   [55.9300, -3.2679],   [51.9187, 4.4957],    [40.6001, -74.3125]],
    'Super':      [[47.0676, 12.5319],   [45.8186, -63.4844],  [41.8610, 12.5368],   [-6.3321, 106.8518],  [35.6062, -77.3732]],
    'Verification':[[48.1787, 16.4149],  [39.3544, -76.4284],  [12.6546, 77.4269],   [53.5362, -113.4709], [65.9409, 12.2172]],
    'Ultra':      [[45.5098, 27.8273],   [41.7107, -93.7364],  [-31.3876, -57.9646], [-37.8981, 144.6260], [54.3667, -6.7719]],
    'Extreme':    [[8.6522, 81.0068],    [46.2477, -80.4337],  [60.3055, 56.9714],   [40.8338, -74.0826],  [43.1914, 17.3802]],
    'Chrome':     [[34.5231, -86.9700],  [52.2930, 4.6669],    [52.5859, -0.2502],   [32.5222, -82.9127],  [39.7692, 30.5314]],
    'World':      [[-6.8147, -38.6534],  [12.1392, -68.9490],  [59.4228, 15.8038],   [51.5530, -0.4759],   [14.3330, 99.6477]],
    'Task':       [[34.2469, -82.2092],  [49.9352, 5.4581],    [43.9436, 12.4477],   [48.0833, -0.6451],   [53.3560, 55.9645]]
}
PREDICTOR_DIR = Path('Roy/ML/Saved_Models')
IMAGE_DIR     = Path('Roy/Test_Images')
EMBED_PATH    = PREDICTOR_DIR / 'geo_embedding_model_r152_normal.pth'

# ----------------- Models -----------------
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        return self.backbone(x).view(x.size(0), -1)

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [2048, 1024, 512, 256, 128, 32, 16]
        for i, (d0, d1) in enumerate(zip(dims, dims[1:])):
            layers += [nn.Linear(d0, d1), nn.BatchNorm1d(d1), nn.GELU(), nn.Dropout(0.1 if i == len(dims)-2 else 0.2)]
        layers.append(nn.Linear(16, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ----------------- Utils -----------------
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_images(testset):
    imgs, paths = [], []
    for fn in sorted(IMAGE_DIR.iterdir()):
        if fn.suffix == '.jpg' and testset in fn.name:
            imgs.append(IMG_TRANSFORM(Image.open(fn).convert('RGB')))
            paths.append(fn)
    return torch.stack(imgs), paths

def haversine(coords1, coords2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [coords1[...,0], coords1[...,1], coords2[...,0], coords2[...,1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ----------------- Pipeline -----------------
def evaluate_testset(testset, device, emb_model, predictor_files, scripted_predictor):
    imgs, paths = load_images(testset)
    imgs = imgs.to(device)
    with torch.no_grad():
        embs = emb_model(imgs)

    all_preds = {}
    with torch.no_grad():
        for fn in predictor_files:
            sd = torch.load(fn, map_location='cpu')
            scripted_predictor.load_state_dict(sd)
            preds = scripted_predictor(embs).cpu().numpy()
            preds[:,0] = (preds[:,0] + 90) % 180 - 90
            preds[:,1] = (preds[:,1] + 180) % 360 - 180
            all_preds[fn.name] = preds

    real = np.array(TEST_SETS[testset])
    mean_err = {name: haversine(real, p).mean() for name, p in all_preds.items()}
    ranked = sorted(mean_err.items(), key=lambda x: x[1])
    return ranked, all_preds, paths, real

# ----------------- Main -----------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_model = GeoEmbeddingModel().to(device).eval()
    emb_model.load_state_dict(torch.load(EMBED_PATH, map_location=device))

    predictor_files = [p for p in PREDICTOR_DIR.iterdir() if p.suffix == '.pth' and 'embedding' not in p.name and 'lowest' not in p.name]
    scripted = torch.jit.script(GeoPredictorNN().to(device).eval())

    for testset in TEST_SETS:
        print(f"\n=== {testset} ===")
        ranked, all_preds, paths, real = evaluate_testset(testset, device, emb_model, predictor_files, scripted)
        for name, err in ranked[:5]:
            print(f"{name}: {err:.2f} km")

        # Plot top model for first image
        top_name = ranked[0][0]
        preds = all_preds[top_name]
        plt.figure(figsize=(6,5))
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.boundary.plot(linewidth=0.5)
        plt.scatter(preds[0,1], preds[0,0], c='r', label='Pred')
        plt.scatter(real[0,1], real[0,0], c='g', label='True')
        plt.title(f"{testset}: {top_name}"); plt.legend(); plt.show()

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Done in {time.time() - start:.1f}s")
