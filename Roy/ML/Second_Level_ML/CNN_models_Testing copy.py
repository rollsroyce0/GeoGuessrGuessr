import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# =========================
# Dataset
# =========================
class CoordinateDataset(Dataset):
    def __init__(self, features, coords):
        self.features = features
        self.coords = coords

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.coords[idx]

# =========================
# Haversine Loss (PyTorch)
# =========================
def haversine_loss_torch(y_true, y_pred):
    """
    y_true, y_pred: (batch_size, 2) in radians
    returns mean Haversine distance in km
    """
    R = 6371.0
    lat1, lon1 = y_true[:,0], y_true[:,1]
    lat2, lon2 = y_pred[:,0], y_pred[:,1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    a = torch.clamp(a, 0.0, 1.0)  # prevent NaNs
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))
    return (R * c).mean()

def haversine_batch(coords1, coords2):
    """NumPy version for scoring"""
    R = 6371.0
    lat1 = np.radians(coords1[:,0]); lon1 = np.radians(coords1[:,1])
    lat2 = np.radians(coords2[:,0]); lon2 = np.radians(coords2[:,1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def geoguessr_points(error):
    return 5000 if error < 0.15 else np.floor(5000 * np.exp(-error/2000))

# =========================
# Feedforward NN
# =========================
class CoordinateRegressor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_nn(model, dataset, epochs, lr, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    print("Starting NN training...")

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            # Convert degrees to radians for Haversine loss
            y_rad = y * (torch.pi / 180)
            pred_rad = pred * (torch.pi / 180)
            loss = haversine_loss_torch(y_rad, pred_rad)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Haversine Loss: {total_loss/len(dataset):.6f}")
    return model

def predict_nn(model, X, device):
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        return model(X).cpu().numpy()

# =========================
# Train/test split
# =========================
def train_test_split(features, coords, test_frac=0.2, seed=42):
    torch.manual_seed(seed)
    dataset = CoordinateDataset(features, coords)
    N = len(dataset)
    test_size = int(N * test_frac)
    train_size = N - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Load CSV
    df = pd.read_csv('Roy/ML/Second_Level_ML/merged_model_predictions_real_coordinates.csv')
    pred_cols = [c for c in df.columns if c.startswith("Predicted_")]
    real_cols = [c for c in df.columns if c.startswith("real_")]

    # Reshape X and y
    X_list, y_list = [], []
    L = len(real_cols)//2  # number of locations per row
    for i in range(len(df)):
        for j in range(L):
            X_list.append(df[[f"Predicted_Latitude{j+1}", f"Predicted_Longitude{j+1}"]].iloc[i].values)
            y_list.append(df[[f"real_latitude{j+1}", f"real_longitude{j+1}"]].iloc[i].values)
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)

    # Train/test split
    train_set, test_set = train_test_split(X, y, test_frac=0.2)
    train_x = torch.stack([x for x, y in train_set])
    train_y = torch.stack([y for x, y in train_set])
    test_x = torch.stack([x for x, y in test_set])
    test_y = torch.stack([y for x, y in test_set])

    # Train NN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model = train_nn(
        CoordinateRegressor(X.shape[1]),
        CoordinateDataset(train_x, train_y),
        epochs=100,
        lr=1e-4,
        batch_size=32,
        device=device
    )

    # Predict
    nn_preds = predict_nn(nn_model, test_x, device)

    # Compute GeoGuessr scores
    nn_scores = [geoguessr_points(haversine_batch(nn_preds[i:i+1], test_y.numpy()[i:i+1])[0])
                 for i in range(len(test_y))]
    print("Average NN Score:", np.mean(nn_scores))

    # Plot predictions vs actual
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(figsize=(15,10), color='white', edgecolor='black')
    plt.scatter(nn_preds[:,1], nn_preds[:,0], color='red', label='Predicted', alpha=0.5)
    plt.scatter(test_y.numpy()[:,1], test_y.numpy()[:,0], color='blue', label='Actual', alpha=0.5)
    plt.legend()
    plt.title('NN Model Predictions vs Actual Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
