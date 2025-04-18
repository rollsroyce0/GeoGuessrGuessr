import os
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from skorch import NeuralNetRegressor

warnings.filterwarnings("ignore")

###########################################
# Utility Function: Extract Coordinates #
###########################################
def extract_coordinates(image_path):
    # Assumes image path format: "<lat>_<lon><...>"
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

###########################################
# Define the Custom Loss (HaversineLoss)  #
###########################################
class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()
        self.earth_radius = 6371.01  # km
        self.pi_on_180 = 0.017453292519943295

    def forward(self, preds, targets):
        # Convert degrees to radians
        lat1 = preds[:, 0] * self.pi_on_180
        lon1 = preds[:, 1] * self.pi_on_180
        lat2 = targets[:, 0] * self.pi_on_180
        lon2 = targets[:, 1] * self.pi_on_180

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a + 1e-7))
        distance = self.earth_radius * c
        return distance.mean()

###########################################
# Define the Coordinate Predictor Network #
###########################################
class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout0 = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(512, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(256, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(128, 32)
        self.batch_norm5 = nn.BatchNorm1d(32)
        self.gelu5 = nn.GELU()
        self.dropout5 = nn.Dropout(0.2)
        
        self.fc6 = nn.Linear(32, 16)
        self.batch_norm6 = nn.BatchNorm1d(16)
        self.gelu6 = nn.GELU()
        self.dropout6 = nn.Dropout(0.1)
        
        self.fc7 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout0(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.gelu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.batch_norm4(x)
        x = self.gelu4(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)  
        x = self.batch_norm5(x)
        x = self.gelu5(x)
        x = self.dropout5(x)
            
        x = self.fc6(x)
        x = self.batch_norm6(x)
        x = self.gelu6(x)
        x = self.dropout6(x)
        
        x = self.fc7(x)
        return x

###########################################
# Load Embeddings and Prepare Data        #
###########################################
# These files should have been generated already.
embeddings_path = 'Roy/ML/embeddings.npy'
image_paths_path = 'Roy/ML/image_paths.npy'

if not os.path.exists(embeddings_path) or not os.path.exists(image_paths_path):
    raise FileNotFoundError("Embeddings or image_paths file not found. Generate embeddings first.")

# Load embeddings and image paths
X = np.load(embeddings_path).astype(np.float32)
image_paths = np.load(image_paths_path)

# Extract coordinates from image_paths
y = np.array([extract_coordinates(p) for p in image_paths])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###########################################
# Wrap the Model using skorch and GridSearchCV #
###########################################
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
net = NeuralNetRegressor(
    module=GeoPredictorNN,
    criterion=HaversineLoss,  # Our custom loss
    max_epochs=150,
    lr=1e-3,
    optimizer=optim.AdamW,
    batch_size=4096,
    device=device_str,
    # Uncomment the following line to see training progress per epoch:
    # verbose=1,
)

# Define a wide hyperparameter grid
param_grid = {
    'lr': [1e-3, 1e-4, 5e-5, 1e-5],
    'optimizer__weight_decay': [0, 1e-4, 1e-5, 1e-6],
    'batch_size': [1024, 2048, 4096, 8192],
    'module__dropout0': [0.1, 0.2, 0.3],
    'module__dropout1': [0.1, 0.2, 0.3],
    'module__dropout2': [0.1, 0.2, 0.3],
    'module__dropout3': [0.1, 0.2, 0.3],
    'module__dropout4': [0.1, 0.2, 0.3],
    'module__dropout5': [0.1, 0.2, 0.3],
    'module__dropout6': [0.05, 0.1, 0.2],
}

gs = GridSearchCV(net, param_grid, refit=True, cv=2, scoring='neg_mean_absolute_error', verbose=2)
gs.fit(X_train, y_train)

print("Best parameters found:")
print(gs.best_params_)
print("Best score (neg MAE):")
print(gs.best_score_)

# Evaluate on the test set
best_net = gs.best_estimator_
test_score = best_net.score(X_test, y_test)
print(f"Test Score (neg MAE): {test_score}")

###########################################
# (Optional) Visualization of Predictions#
###########################################
def predict_coordinates(embedding, model):
    model.module_.eval()
    with torch.no_grad():
        emb_tensor = torch.tensor(embedding).float().unsqueeze(0).to(device_str)
        pred = model.module_(emb_tensor).cpu().numpy()[0]
    return pred

# Plot a few predictions vs. true coordinates
n_samples = 100
y_pred = np.array([predict_coordinates(embedding, best_net) for embedding in X_test[:n_samples]])

plt.figure(figsize=(10, 8))
plt.scatter(y_test[:n_samples, 1], y_test[:n_samples, 0], color='blue', label='True')
plt.scatter(y_pred[:, 1], y_pred[:, 0], color='red', label='Predicted')
for i in range(n_samples):
    plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='green', alpha=0.5)
plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
