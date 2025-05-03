import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.linear_model import LinearRegression
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
            dropout_rate = 0.1 if i == len(dims)-2 else 0.2
            setattr(self, f'dropout{i+1}', nn.Dropout(dropout_rate))
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    images, img_paths = load_images('Roy/Test_Images', testtype)
    images = images.to(device)

    real_coords_dict = {
        'Game': [[59.2642, 10.4276], [1.4855, 103.8676], [54.9927, -1.6732], [19.4745, -99.1974], [58.6133, 49.6275]],
        'Validation': [[43.3219114, -5.5783907], [23.0376137, 72.5819308], [55.9300025, -3.2678762], [51.9187417, 4.4957128], [40.6000729, -74.3125485]],
        'Verification': [[48.1787242,16.4149478], [39.3544037,-76.4284282], [12.6545729,77.4269159], [53.5361597,-113.470894], [65.9408919,12.2171864]],
        'Super': [[47.0676173,12.5318788], [45.8186432,-63.4844332], [41.8610051,12.5368213], [-6.3320979,106.8518361], [35.6061998,-77.3731937]],
        'Ultra': [[45.5097937,27.8273201], [41.71066,-93.7363551], [-31.387591,-57.9646316], [-37.8980561,144.626041], [54.3667423,-6.7718667]],
        'Validation': [[43.3219114, -5.5783907], [23.0376137, 72.5819308], [55.9300025, -3.2678762], [51.9187417, 4.4957128], [40.6000729, -74.3125485]],
        'Verification': [[48.1787242,16.4149478], [39.3544037,-76.4284282], [12.6545729,77.4269159], [53.5361597,-113.470894], [65.9408919,12.2171864]],
    }

    real_coords = np.array(real_coords_dict[testtype])
    embed_model = GeoEmbeddingModel().to(device).eval()
    embed_model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth', map_location=device))

    with torch.no_grad():
        embeddings = embed_model(images).cpu()

    model_preds = []
    for fname in sorted(os.listdir('Roy/ML/Saved_Models')):
        if 'embedding' in fname or 'lowest' in fname or not fname.endswith('.pth'):
            continue
        predictor = GeoPredictorNN().to(device).eval()
        predictor.load_state_dict(torch.load(f'Roy/ML/Saved_Models/{fname}', map_location=device))
        with torch.no_grad():
            preds = predictor(embeddings.to(device)).cpu().numpy()
            preds[:,0] = (preds[:,0]+90)%180 - 90
            preds[:,1] = (preds[:,1]+180)%360 - 180
            model_preds.append((fname, preds))

    return real_coords, model_preds

# -------- Meta Training --------
def train_meta_model():
    meta_features, meta_targets, model_names = [], [], []
    testtypes = ['Game', 'Validation', 'Super', 'Verification']

    for tt in testtypes:
        real_coords, model_preds = main(tt)
        for i, (fname, preds) in enumerate(model_preds):
            for j in range(len(real_coords)):
                meta_features.append([*preds[j], i, j])
                meta_targets.append(real_coords[j])
                model_names.append(fname)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    name_encoded = encoder.fit_transform(np.array(model_names).reshape(-1,1))

    X = np.hstack([np.array(meta_features), name_encoded])
    y = np.array(meta_targets)

    reg = LinearRegression()
    reg.fit(X, y)
    joblib.dump((reg, encoder), 'Roy/ML/Saved_Models/meta_model.pkl')
    print("Meta-model trained and saved.")
    return reg, encoder

# -------- Prediction Using Meta Model --------
def predict_coords(meta_model, encoder, model_preds):
    reg, enc = meta_model
    all_preds = []
    for j in range(len(model_preds[0][1])):
        X = []
        for i, (fname, preds) in enumerate(model_preds):
            base_feats = [*preds[j], i, j]
            name_enc = enc.transform([[fname]])
            X.append(np.hstack([base_feats, name_enc[0]]))
        X = np.array(X)
        preds_for_image = reg.predict(X)
        all_preds.append(preds_for_image.mean(axis=0))
    return np.array(all_preds)

# -------- Main Block --------
if __name__ == "__main__":
    start_time = time.time()
    reg_model, encoder = train_meta_model()

    # Predict on a new test type using meta model
    testtype = 'Validation'
    real_coords, model_preds = main(testtype)
    pred_coords = predict_coords((reg_model, encoder), encoder, model_preds)

    print("Predicted coordinates:\n", pred_coords)
    print("Real coordinates:\n", real_coords)

    mse = mean_squared_error(real_coords, pred_coords)
    print(f"MSE: {mse:.2f}")

    plt.scatter(real_coords[:,0], real_coords[:,1], label='True', alpha=0.7)
    plt.scatter(pred_coords[:,0], pred_coords[:,1], label='Predicted', alpha=0.7)
    plt.legend(); plt.xlabel("Lat"); plt.ylabel("Lon"); plt.title("Meta-Model Predictions")
    plt.grid(); plt.show()
    print(f"Execution time: {time.time() - start_time:.2f}s")
