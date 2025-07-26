import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import joblib

warnings.filterwarnings("ignore")

# -------- Models --------
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [2048, 1024, 512, 256, 128, 32, 16]
        for i in range(len(dims)-1):
            in_dim, out_dim = dims[i], dims[i+1]
            setattr(self, f'fc{i+1}', nn.Linear(in_dim, out_dim))
            setattr(self, f'batch_norm{i+1}', nn.BatchNorm1d(out_dim))
            setattr(self, f'gelu{i+1}', nn.GELU())
            dr = 0.1 if i == len(dims)-2 else 0.2
            setattr(self, f'dropout{i+1}', nn.Dropout(dr))
        self.fc7 = nn.Linear(16, 2)

    def forward(self, x):
        for i in range(1, 7):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'dropout{i}')(x)
            x = getattr(self, f'batch_norm{i}')(x)
            x = getattr(self, f'gelu{i}')(x)
        return self.fc7(x)

# -------- Utils --------
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_images(folder, testtype):
    imgs, paths = [], []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith('.jpg') and testtype in fn:
            img = Image.open(os.path.join(folder, fn)).convert('RGB')
            imgs.append(transform(img))
            paths.append(os.path.join(folder, fn))
    return torch.stack(imgs), paths

# -------- Main Evaluation Function --------
def main(testtype):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images, _ = load_images('Roy/Test_Images', testtype)
    images = images.to(device)

    real_coords_dict = {
        'Game':       [[59.2642, 10.4276], [1.4855, 103.8676], [54.9927, -1.6732], [19.4745, -99.1974], [58.6133, 49.6275]],
        'Validation': [[43.3219, -5.5784],   [23.0376, 72.5819],   [55.9300, -3.2679],   [51.9187, 4.4957],    [40.6001, -74.3125]],
        'Verification':[[48.1787, 16.4149],[39.3544, -76.4284],[12.6546, 77.4269],[53.5362,-113.4709],[65.9409,12.2172]],
        'Super':      [[47.0676, 12.5319],[45.8186, -63.4844],[41.8610, 12.5368],[-6.3321, 106.8518],[35.6062, -77.3732]],
        'Ultra':      [[45.5098,27.8273],[41.7107,-93.7364],[-31.3876,-57.9646],[-37.8981,144.6260],[54.3667,-6.7719]],
        'Extreme':    [[8.6522,81.0068],[46.2477,-80.4337],[60.3055,56.9714],[40.8338,-74.0826],[43.1914,17.3802]],
        'Chrome':     [[34.5231,-86.9700],[52.2930,4.6669],[52.5859,-0.2502],[32.5222,-82.9127],[39.7692,30.5314]],
        'World':      [[-6.8147,-38.6534],[12.1392,-68.9490],[59.4228,15.8038],[51.5530,-0.4759],[14.3330,99.6477]],
        'Task':       [[34.2469,-82.2092],[49.9352,5.4581],[43.9436,12.4477],[48.0833,-0.6451],[53.3560,55.9645]],
        'Enlarged':   [[-34.8295223,-58.8707693], [40.4369798,-3.6859228], [-54.1257734,-68.0709486], [48.9828428,12.6387341], [45.9312686,-82.4707373]],
        'Exam':       [[-4.1237242,-38.3705862], [40.1161881,-75.1248975], [35.1362241,136.7419345], [41.6557297,-91.5466424], [-47.0777189,-72.1646972]],
        'Google':    [[59.407269,15.415694], [52.5644145,-110.8206357], [-36.8700509,174.6481411], [37.9270951,-122.53026], [28.6397445,77.2929918]]
    }

    real_coords = np.array(real_coords_dict[testtype])
    # Embedding
    embed_model = GeoEmbeddingModel().to(device).eval()
    embed_model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth', map_location=device))
    with torch.no_grad():
        embeddings = embed_model(images).cpu().numpy()

    # Per-model predictions
    model_preds = []
    for fname in sorted(os.listdir('Roy/ML/Saved_Models')):
        if 'embedding' in fname or 'lowest' in fname or not fname.endswith('.pth'): continue
        predictor = GeoPredictorNN().to(device).eval()
        predictor.load_state_dict(torch.load(f'Roy/ML/Saved_Models/{fname}', map_location=device))
        with torch.no_grad():
            p = predictor(torch.from_numpy(embeddings).to(device)).cpu().numpy()
            p[:,0] = (p[:,0]+90)%180 - 90
            p[:,1] = (p[:,1]+180)%360 - 180
        model_preds.append((fname, p))
    return real_coords, model_preds

# -------- Meta Training with Gaussian Process --------
def train_meta_model():
    feats, targets, names = [], [], []
    splits = ['Game','Super','Verification','Ultra','Extreme','Chrome','World','Task', 'Google', 'Enlarged', 'Exam']
    for tt in splits:
        real_coords, preds = main(tt)
        for mid, (fname, p) in enumerate(preds):
            for idx in range(len(real_coords)):
                feats.append([p[idx,0], p[idx,1], mid, idx])
                targets.append(real_coords[idx])
                names.append(fname)

    # Encode model names
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    name_enc = enc.fit_transform(np.array(names).reshape(-1,1))

    X = np.hstack([np.array(feats), name_enc])
    y = np.array(targets)

    # Gaussian Process with RBF kernel
    print("Training meta-model (Gaussian Process)...")
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-2, normalize_y=True)
    gp.fit(X, y)

    joblib.dump((gp, enc), 'Roy/ML/Saved_Models/meta_model_gaussian.pkl')
    print("Meta-model (Gaussian Process) trained and saved.")
    return gp, enc

# -------- Predict with Meta Model --------
def predict_coords(meta, enc, model_preds):
    rf = meta
    allp = []
    for img_i in range(model_preds[0][1].shape[0]):
        rows = []
        for mid, (fname,p) in enumerate(model_preds):
            base = [p[img_i,0], p[img_i,1], mid, img_i]
            nm = enc.transform([[fname]])[0]
            rows.append(base + nm.tolist())
        preds = rf.predict(np.array(rows))
        allp.append(preds.mean(axis=0))
    return np.array(allp)

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # Radius of the Earth in kilometers
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c  # Distance in kilometers

# -------- Main Block --------
if __name__ == "__main__":
    start = time.time()
    rf_meta, encoder = train_meta_model()

    # Evaluate on one split
    testtype = 'Validation'  # Change this to 'Game', 'Super', 'Verification', 'Ultra', 'Extreme', 'Chrome', 'World', 'Task', 'Google', 'Enlarged', 'Exam'
    real_coords, preds = main(testtype)
    pred_coords = predict_coords(rf_meta, encoder, preds)
    
    errors = []
    for i in range(len(real_coords)):
        dist = haversine(real_coords[i,1], real_coords[i,0], pred_coords[i,1], pred_coords[i,0])
        errors.append(dist)
    print("Mean error (km):", np.mean(errors))
    print("Max error (km):", np.max(errors))
    print("Min error (km):", np.min(errors))
    print("Median error (km):", np.median(errors))
    print("Std error (km):", np.std(errors))

    print("Real:      ", real_coords)
    print("Predicted: ", pred_coords)
    print("MSE:", mean_squared_error(real_coords, pred_coords))

    plt.scatter(real_coords[:,0], real_coords[:,1], label='True', alpha=0.7)
    plt.scatter(pred_coords[:,0], pred_coords[:,1], label='Pred', alpha=0.7)
    plt.legend(); plt.xlabel("Lat"); plt.ylabel("Lon"); plt.title("Meta-RF Predictions")
    plt.grid(); plt.show()

    print(f"Total time: {time.time() - start:.1f}s")
