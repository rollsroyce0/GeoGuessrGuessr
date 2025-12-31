import pandas as pd 
import numpy as np 
import math 
import geopandas as gpd 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import gpytorch
import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from Enumerate_all_models import enumerate_models 
from generate_coordinates import main as main_coords 
from Merge_dfs  import main as main_merge 
from Generate_model_predictions  import main as main_predictions 
# ========================= 
# # Utils (run if new data was acquired) 
# # ========================= 
if True: 
    main_predictions() 
    main_coords() 
    main_merge()

# =========================
# Load data
# =========================

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    return df

def model_indexing(df): #TODO: Maybe remove ones that are not on the leaderboard 
    leaderboard_only=True
    model_files = enumerate_models('Roy/ML/Saved_Models/', leaderboard_only=leaderboard_only)
    print(f"Indexing {len(model_files)} models from enumerated files")
    model_name_to_index = {name: idx for idx, name in enumerate(model_files)}
    print(f"Indexing {len(model_name_to_index)} models")
    df['Model_Index'] = df['Model_Name'].map(model_name_to_index)
    df = df.dropna(subset=['Model_Index'])
    df['Model_Index'] = df['Model_Index'].astype(int)

    return df

def panda_tester(df):
    print("Panda tester function")
    print(df.head())
    print(df.columns)
    print(df.dtypes)
    print(df.describe())
    print(df.info())
    print("Unique models:", df['Model_Name'].nunique())
    print("Unique test types:", df['test_type'].unique())
    return df

# =========================
# Dataset class
# =========================

class CoordinateDataset(Dataset):
    def __init__(self, features, coords):
        """
        features: torch.Tensor of shape (N, D)
        coords: torch.Tensor of shape (N, 2) -> (lat, lon in radians)
        """
        self.features = features
        self.coords = coords

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.coords[idx]

# =========================
# Metrics
# =========================

def haversine_km_np(pred, target):
    """
    pred, target: (2,) or (..., 2) in degrees
    returns: km
    """
    pred = torch.tensor(pred, dtype=torch.float64)
    target = torch.tensor(target, dtype=torch.float64)

    lat1, lon1 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    lat2, lon2 = torch.deg2rad(target[..., 0]), torch.deg2rad(target[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return float(6371.0 * c)


def geoguessr_score(km):
    return 5000 if km < 0.15 else int(np.floor(5000 * np.exp(-km / 2000)))


# ===========================
# Plotting
# ===========================
def plot_coordinates_on_map(pred, real, backups, path, avg):
    plt.figure(figsize=(10,8))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=plt.gca(), linewidth=1)
    for b in backups:
        plt.scatter(b[1], b[0], c='blue', s=40, alpha=0.25)
    plt.scatter(pred[1], pred[0], c='red', label='Predicted', s=100)
    plt.scatter(avg[1], avg[0], c='orange', label='Averaged', s=100)
    plt.scatter(real[1], real[0], c='green', label='Real', s=100, alpha=1)
    
    allc = np.vstack([*backups, pred, real])
    lat_min, lat_max = allc[:,0].min(), allc[:,0].max()
    lon_min, lon_max = allc[:,1].min(), allc[:,1].max()
    mlat = (lat_max - lat_min)*0.5+2; mlon = (lon_max - lon_min)*0.5+2
    plt.xlim(lon_min-mlon, lon_max+mlon); plt.ylim(lat_min-mlat, lat_max+mlat)
    plt.title(f"Map: {os.path.basename(path)}"); plt.xlabel('Lon'); plt.ylabel('Lat')
    plt.legend(); plt.show(block=False); plt.pause(5)
    plt.close()

# ---------------------------
#  Haversine Loss
# ---------------------------
def haversine_loss(y_true, y_pred):
    """
    y_true, y_pred: (batch_size, 2), lat/lon in radians
    returns mean Haversine distance in radians
    """
    dlat = y_pred[:, 0] - y_true[:, 0]
    dlon = y_pred[:, 1] - y_true[:, 1]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(y_true[:, 0]) * torch.cos(y_pred[:, 0]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return torch.mean(c)

# ---------------------------
#  1) Feedforward Regression NN
# ---------------------------
class CoordinateRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # output: lat, lon in radians
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
#  2) Gaussian Process Regression (GPyTorch)
# ---------------------------
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2])
        )

    def forward(self, x):
        mean_x = self.mean_module(x).transpose(-1, -2)  # shape [batch, 2]
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
# =========================
# Evaluation training functions
# =========================

def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            loss = haversine_loss(y, pred)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(dataset)

# ---------------------------
#  Training function for NN
# ---------------------------
def train_nn(model, dataset, epochs=50, lr=1e-3, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = haversine_loss(y, pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Haversine Loss: {total_loss/len(dataset):.6f}")
    return model
        
def train_gp(train_x, train_y, epochs=50, lr=0.1):
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = GPRegressionModel(train_x, train_y, likelihood)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.transpose(-1, -2))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
    return model, likelihood


def train(model_type, train_x, train_y, epochs=50):
    if model_type == 'nn':
        dataset = CoordinateDataset(train_x, train_y)
        model = CoordinateRegressor(input_dim=train_x.shape[1])
        model = train_nn(model, dataset, epochs=epochs)
        return model
    elif model_type == 'gp':
        model, likelihood = train_gp(train_x, train_y, epochs=epochs)
        return model, likelihood
    else:
        raise ValueError("Unknown model type")
    



# =========================
# Averaging baseline
# =========================

def average_baseline(df, report_score=True):
    """
    For each test_type and each location:
    - average all model predictions
    - compute Haversine error
    """
    
    df = model_indexing(df)
    print(f"Data after indexing has {len(df)} rows")
    print(df.head())
    panda_tester(df)


    pred_cols = [c for c in df.columns if c.startswith("Predicted_")]
    real_cols = [c for c in df.columns if c.startswith("real_")]
    print(f"Found {len(pred_cols)} prediction columns and {len(real_cols)} real columns")

    L = len(pred_cols) // 2  # number of locations (should be 5)

    results = {}

    for test_type, g in df.groupby("test_type"):
        print(f"\n=== {test_type} ===")
        loc_errors = []
        loc_errors_avg = []

        for loc in range(L):
            plat = f"Predicted_Latitude{loc+1}"
            plon = f"Predicted_Longitude{loc+1}"
            rlat = f"real_latitude{loc+1}"
            rlon = f"real_longitude{loc+1}"
            #print(f"\nLocation {loc+1}: {plat}, {plon} vs {rlat}, {rlon}")
            #print(g)
            preds = g[[plat, plon]].to_numpy()     # (num_models, 2)
            real = g[[rlat, rlon]].iloc[0].to_numpy()
            print(f"  Number of model predictions: {preds.shape[0]}")

            ML_pred = np.mean(preds, axis=0)
            
            avg_pred = np.mean(preds, axis=0)

            km = haversine_km_np(ML_pred, real)
            km_avg = haversine_km_np(avg_pred, real)
            loc_errors.append(km)
            loc_errors_avg.append(km_avg)

            if report_score:
                score = geoguessr_score(km)
                score_avg = geoguessr_score(km_avg)
                print(f"  Location {loc+1}: {km:7.1f} km | score {score} | Avg: {km_avg:7.1f} km | score {score_avg}")
            else:
                print(f"  Location {loc+1}: {km:7.1f} km")
            # Optional: plot
            if False:
                backups = preds.tolist()
                plot_coordinates_on_map(reg_pred, real, backups, path=f"Reg_pred_{test_type}_loc{loc+1}.png", avg=avg_pred)

        mean_km = np.mean(loc_errors)
        mean_km_avg = np.mean(loc_errors_avg)
        if report_score:
            mean_score = geoguessr_score(mean_km)
            mean_score_avg = geoguessr_score(mean_km_avg)
            print(f"\n  Mean over {L} locations: {mean_km:7.1f} km | score {mean_score} | Avg: {mean_km_avg:7.1f} km | score {mean_score_avg}")

        

        results[test_type] = {
            "per_location_km": loc_errors,
            "mean_km": mean_km,
            "mean_score": mean_score
        }

    return results


# =========================
# Optional diagnostics
# =========================

def per_model_baseline(df):
    """
    Computes mean km error per model across all test_types and locations.
    """
    pred_cols = [c for c in df.columns if c.startswith("Predicted_")]
    real_cols = [c for c in df.columns if c.startswith("real_")]
    L = len(pred_cols) // 2

    rows = []

    for model_name, g in df.groupby("Model_Name"):
        errs = []
        for _, row in g.iterrows():
            for loc in range(L):
                pred = row[[f"Predicted_latitude{loc+1}", f"Predicted_longitude{loc+1}"]].values
                real = row[[f"real_latitude{loc+1}", f"real_longitude{loc+1}"]].values
                errs.append(haversine_km_np(pred, real))

        rows.append((model_name, np.mean(errs)))

    res = pd.DataFrame(rows, columns=["Model", "Mean_km"]).sort_values("Mean_km")
    print("\n=== Per-model mean error ===")
    print(res)
    return res


# =========================
# Main
# =========================

def main():
    csv = "Roy/ML/Second_Level_ML/merged_model_predictions_real_coordinates.csv"
    df = load_data(csv)

    print("\nRunning average-ensemble baseline …")
    results = average_baseline(df, report_score=True)

    # Optional: uncomment for diagnostics
    # per_model_baseline(df)


if __name__ == "__main__":
    main()
