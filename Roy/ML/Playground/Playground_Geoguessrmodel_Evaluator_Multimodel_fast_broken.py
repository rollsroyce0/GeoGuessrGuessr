import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import geopandas as gpd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

# ------------------------
# Configurations and Paths
# ------------------------
DATA_DIR = Path('Roy/Test_Images')
MODEL_DIR = Path('Roy/ML/Saved_Models')
EMBED_MODEL_PATH = MODEL_DIR / 'geo_embedding_model_r152_normal.pth'
OUTPUT_DIR = DATA_DIR

TEST_TYPES = [
    'Game', 'Validation', 'Verification', 'Super', 'Ultra',
    'Extreme', 'Chrome', 'World', 'Task', 'Enlarged',
    'Exam', 'Google'
]

REAL_COORDS = {
    'Game':        np.array([[59.2642, 10.4276], [1.4855, 103.8676], [54.9927, -1.6732], [19.4745, -99.1974], [58.6133, 49.6275]]),
    'Validation':  np.array([[43.3219114, -5.5783907], [23.0376137, 72.5819308], [55.9300025, -3.2678762], [51.9187417, 4.4957128], [40.6000729, -74.3125485]]),
    'Verification':np.array([[48.1787242, 16.4149478], [39.3544037, -76.4284282], [12.6545729, 77.4269159], [53.5361597, -113.470894], [65.9408919, 12.2171864]]),
    'Super':       np.array([[47.0676173, 12.5318788], [45.8186432, -63.4844332], [41.8610051, 12.5368213], [-6.3320979, 106.8518361], [35.6061998, -77.3731937]]),
    'Ultra':       np.array([[45.5097937, 27.8273201], [41.71066, -93.7363551], [-31.387591, -57.9646316], [-37.8980561, 144.626041], [54.3667423, -6.7718667]]),
    'Extreme':     np.array([[8.6521503, 81.0067919], [46.2477428, -80.4337085], [60.3055298, 56.971439], [40.8337969, -74.0825815], [43.1914376, 17.3801763]]),
    'Chrome':      np.array([[34.5230872, -86.9699949], [52.2929603, 4.6668573], [52.585936, -0.2501907], [32.5221938, -82.9127378], [39.7692443, 30.5314142]]),
    'World':       np.array([[-6.8146562, -38.6533882], [12.1391977, -68.9490383], [59.4227739, 15.8038038], [51.5529906, -0.4758671], [14.3329551, 99.6477487]]),
    'Task':        np.array([[34.2468633, -82.2092303], [49.935202, 5.4581067], [43.9435807, 12.4477353], [48.08332, -0.6451421], [53.3559593, 55.9645235]]),
    'Enlarged':    np.array([[-34.8295223, -58.8707693], [40.4369798, -3.6859228], [-54.1257734, -68.0709486], [48.9828428, 12.6387341], [45.9312686, -82.4707373]]),
    'Exam':        np.array([[-4.1237242, -38.3705862], [40.1161881, -75.1248975], [35.1362241, 136.7419345], [41.6557297, -91.5466424], [-47.0777189, -72.1646972]]),
    'Google':      np.array([[59.407269, 15.415694], [52.5644145, -110.8206357], [-36.8700509, 174.6481411], [37.9270951, -122.53026], [28.6397445, 77.2929918]])
}

# ------------------------
# Preload heavy resources
# ------------------------
WORLD = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# Models Definition
# ------------------------
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.backbone(x).flatten(1)

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [2048, 1024, 512, 256, 128, 32, 16]
        for i in range(len(dims)-1):
            in_dim, out_dim = dims[i], dims[i+1]
            setattr(self, f'fc{i+1}', nn.Linear(in_dim, out_dim))
            setattr(self, f'batch_norm{i+1}', nn.BatchNorm1d(out_dim))
            setattr(self, f'gelu{i+1}', nn.GELU())
            # smaller dropout on last block
            dropout_rate = 0.1 if i == len(dims)-2 else 0.2
            setattr(self, f'dropout{i+1}', nn.Dropout(dropout_rate))
        # final layer
        self.fc7 = nn.Linear(16, 2)

    def forward(self, x):
        # sequentially apply each block
        for i in range(1, 7):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'dropout{i}')(x)
            x = getattr(self, f'batch_norm{i}')(x)
            x = getattr(self, f'gelu{i}')(x)
        x = self.fc7(x)
        return x

# ------------------------
# Dataset & Transforms
# ------------------------
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, folder, keyword, transform):
        self.paths = sorted(folder.glob(f'*{keyword}*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), str(self.paths[idx])

# ------------------------
# Utility Functions
# ------------------------
def haversine(coords1, coords2):
    R = 6371.0
    lat1, lon1 = np.radians(coords1[:,0]), np.radians(coords1[:,1])
    lat2, lon2 = np.radians(coords2[:,0]), np.radians(coords2[:,1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def geoguessr_points(err):
    return np.where(err<0.15, 5000, np.floor(5000*np.exp(-err/2000)))

# ------------------------
# Evaluation Function
# ------------------------
# Embedding model loaded once
embed_model = GeoEmbeddingModel().to(DEVICE).eval()
torch.load(EMBED_MODEL_PATH, map_location=DEVICE)
embed_model.load_state_dict(torch.load(EMBED_MODEL_PATH, map_location=DEVICE))

def evaluate(testtype: str):
    # Dataset loader with parallel workers
    dataset = ImageDataset(DATA_DIR, testtype, transform)
    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    # Precompute embeddings in batches
    embeddings, paths = [], []
    with torch.no_grad():
        for batch, pth in loader:
            batch = batch.to(DEVICE)
            embeddings.append(embed_model(batch).cpu())
            paths.extend(pth)
    embeddings = torch.cat(embeddings, dim=0)

    # Predictor checkpoints
    predictors = sorted([p for p in MODEL_DIR.glob('*.pth')
                         if 'embedding' not in p.name and 'lowest' not in p.name])

    results, errs_stack = [], []
    # Iterate predictors
    for ckpt in predictors:
        model = GeoPredictorNN().to(DEVICE).eval()
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        # JIT optimize once
        model = torch.jit.script(model)

        with torch.no_grad():
            preds = model(embeddings.to(DEVICE)).cpu().numpy()
        preds[:,0] = ((preds[:,0]+90)%180) - 90
        preds[:,1] = ((preds[:,1]+180)%360) - 180

        err = haversine(REAL_COORDS[testtype], preds)
        pts = geoguessr_points(err)
        results.append((ckpt.name, int(pts.sum()), preds))
        errs_stack.append(err)

    # Select top10
    top10 = sorted(results, key=lambda x: x[1], reverse=True)[:10]

    # Write best models
    out_txt = OUTPUT_DIR / f'Best_models_{testtype}.txt'
    out_txt.write_text(f"{testtype}: {', '.join(name for name,_,_ in top10)}")

    # Ensemble average & final points
    avg_pred = np.mean(np.stack([p for *_, p in top10]), axis=0)
    final_err = haversine(REAL_COORDS[testtype], avg_pred)
    final_pts = geoguessr_points(final_err)

    # Difficulty scoring
    errs_arr = np.stack(errs_stack, axis=0)
    diff_score = np.round((np.log10((errs_arr.std(0)+0.4*errs_arr.mean(0))/1000 + 1)*13).clip(0,10)**2,3)
    diff_out = OUTPUT_DIR / 'Difficulty_scores.txt'
    lines = diff_out.read_text().splitlines() if diff_out.exists() else []
    new_line = f"{testtype}: {diff_score.mean():.3f}, Highest: {diff_score.max():.3f}, Lowest: {diff_score.min():.3f}"
    lines = [l for l in lines if not l.startswith(testtype+':')] + [new_line]
    diff_out.write_text('\n'.join(lines))

    # Plotting reuse global WORLD
    for pred, real, path in zip(avg_pred, REAL_COORDS[testtype], paths):
        plt.figure(figsize=(8,6))
        WORLD.boundary.plot(linewidth=0.5)
        plt.scatter(*pred[::-1], c='red', label='Pred', s=30)
        plt.scatter(*real[::-1], c='green', label='Real', s=30, alpha=0.6)
        plt.axis('off')
        plt.title(Path(path).name)
        plt.show()

    return top10, final_pts, diff_score

# ------------------------
# Main Execution
# ------------------------
def main():
    start = time.time()
    for t in TEST_TYPES:
        print(f"Evaluating {t}...")
        top, pts, diff = evaluate(t)
        print(f"{t} -> Total Points: {pts.sum()}, Difficulty Avg: {diff.mean():.3f}")
    print(f"Completed in {time.time()-start:.2f}s")

if __name__ == '__main__':
    main()
