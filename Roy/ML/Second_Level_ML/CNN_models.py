import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Enumerate_all_models import enumerate_models
from generate_coordinates import main as main_coords
from Merge_dfs import main as main_merge
from Generate_model_predictions import main as main_predictions


# =========================
# Utils
# =========================
if False:
    main_predictions()
    main_coords()
    main_merge()

# =========================
# Data
# =========================

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    return df

def Geoguessr_score(error):
    return 5000 if error < 0.15 else np.floor(5000 * np.exp(-error/2000))

def model_indexing(df):
    enumerate_models('Roy/ML/Saved_Models/')
    model_files = pd.read_csv('Roy/ML/Second_Level_ML/model_files.csv')
    model_list = model_files['model_file'].tolist()
    model_index = {m: i for i, m in enumerate(model_list)}
    df['Model_Index'] = df['Model_Name'].map(model_index)
    return df


def build_cnn_dataset(df):
    y_cols = [
        'real_latitude1','real_longitude1',
        'real_latitude2','real_longitude2',
        'real_latitude3','real_longitude3',
        'real_latitude4','real_longitude4',
        'real_latitude5','real_longitude5'
    ]

    feature_cols = [
        c for c in df.columns
        if c not in ['test_type', 'Model_Name', 'Model_Index'] + y_cols
    ]

    X, y = [], []

    for _, g in df.groupby('test_type'):
        g = g.sort_values('Model_Index')

        X.append(torch.tensor(
            g[feature_cols].values, dtype=torch.float32
        ))

        y.append(torch.tensor(
            g[y_cols].iloc[0].values, dtype=torch.float32
        ))

    X = torch.stack(X)
    y = torch.stack(y)

    print(f"Built dataset: X {X.shape}, y {y.shape}")
    return X, y


def train_split_data(df, split=0.9):
    df = model_indexing(df)
    X, y = build_cnn_dataset(df)

    idx = torch.randperm(len(X))
    s = int(split * len(X))

    return (
        X[idx[:s]],
        y[idx[:s]],
        X[idx[s:]],
        y[idx[s:]]
    )


# =========================
# Model
# =========================

class CNNEnsemble(nn.Module):
    def __init__(self, num_models, hidden=64):
        super().__init__()

        self.conv1 = nn.Conv1d(num_models, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, num_models),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden * 10, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))

        w = self.attn(h).unsqueeze(-1)
        h = h * w.sum(dim=1, keepdim=True)

        return self.fc(h.flatten(1))


# =========================
# Loss
# =========================

def haversine_loss(pred, target):
    pred = pred.view(-1, 5, 2)
    target = target.view(-1, 5, 2)

    lat1, lon1 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    lat2, lon2 = torch.deg2rad(target[..., 0]), torch.deg2rad(target[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return (6371 * c).mean()


# =========================
# Training
# =========================

def train(model, Xtr, ytr, Xte, yte, epochs=2000, lr=4e-6):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epochs):
        model.train()
        loss = haversine_loss(model(Xtr), ytr)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val = haversine_loss(model(Xte), yte)
        if e % 100 == 0:
            print(f"{e+1:03d} | train {loss:.2f} km | val {val:.2f} km | lr {lr:.1e} | Points {Geoguessr_score(loss.item()):.1f} | Val Points {Geoguessr_score(val.item()):.1f}")


# =========================
# Main
# =========================

def main():
    csv = 'Roy/ML/Second_Level_ML/merged_model_predictions_real_coordinates.csv'
    df = load_data(csv)

    Xtr, ytr, Xte, yte = train_split_data(df)

    num_models = Xtr.shape[1]
    print(f"Number of models: {num_models}")

    model = CNNEnsemble(num_models)
    train(model, Xtr, ytr, Xte, yte)

    torch.save({
        'state_dict': model.state_dict(),
        'num_models': num_models
    }, 'cnn_geo_ensemble.pt')

    print("Model saved → cnn_geo_ensemble.pt")


if __name__ == "__main__":
    main()
