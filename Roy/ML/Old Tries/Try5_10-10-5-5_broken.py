import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor as Decision
from sklearn.metrics import mean_squared_error
import warnings
import geopandas as gpd
from shapely.geometry import Point
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

# Function to extract coordinates from image path
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the location of the images
location = "D:/GeoGuessrGuessr/geoguesst"

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class GeoDataset(Dataset):
    def __init__(self, image_paths, coordinates, transform=None):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        coords = self.coordinates[idx]
        return image, coords

# Custom Model
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super(GeoEmbeddingModel, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final classification layer
        self.fc = nn.Linear(2048, 4096)  # Add a fully connected layer

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load or generate embeddings
if os.path.exists('Roy/ML/embeddings.npy'):
    embeddings = np.load('Roy/ML/embeddings.npy').astype(np.float64)  # Ensure embeddings are float64
    image_paths = np.load('Roy/ML/image_paths.npy')
else:
    # Load image paths and extract coordinates
    image_paths = [os.path.join(location, img_file) for img_file in os.listdir(location)]
    coordinates = np.array([extract_coordinates(path) for path in image_paths])

    # Initialize the custom model
    model = GeoEmbeddingModel().to(device)
    
    # Dataset and DataLoader
    dataset = GeoDataset(image_paths=image_paths, coordinates=coordinates, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the pre-trained model if available
    if os.path.exists('geo_embedding_model.pth'):
        model.load_state_dict(torch.load('geo_embedding_model.pth'))

    # Generate embeddings
    print("Generating custom embeddings...")
    embeddings = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, _ in track(dataloader, description="Processing images..."):
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy().astype(np.float64))  # Move to CPU and convert to float64
    embeddings = np.vstack(embeddings)

    # Save the embeddings and image paths
    np.save('Roy/ML/embeddings.npy', embeddings)
    np.save('Roy/ML/image_paths.npy', np.array(image_paths))

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")

# Extract the latitude and longitude from each image path
coordinates = np.array([extract_coordinates(path) for path in image_paths])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, coordinates, test_size=5000/embeddings.shape[0])

# Step 1: Multi-Level Clustering (10-10-5-5)
def multi_level_clustering(X, n_clusters):
    """Apply a multi-level clustering on the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

# Dictionary to hold multi-level cluster models
cluster_hierarchy = {}
model_hierarchy = {}

print("Performing multi-level clustering and training models...")
current_X = X_train
current_labels = None

# Level 1: 10 Clusters
kmeans1, labels1 = multi_level_clustering(current_X, 10)
cluster_hierarchy[1] = (kmeans1, labels1)

for i in track(range(10)):
    cluster_data1 = current_X[labels1 == i]
    cluster_coords1 = y_train[labels1 == i]
    
    # Level 2: 10 Sub-Clusters within each of the 10 clusters from Level 1
    kmeans2, labels2 = multi_level_clustering(cluster_data1, 10)
    cluster_hierarchy[(1, i)] = (kmeans2, labels2)
    
    for j in range(10):
        cluster_data2 = cluster_data1[labels2 == j]
        cluster_coords2 = cluster_coords1[labels2 == j]
        
        # Level 3: 5 Sub-Clusters within each of the 10 sub-clusters from Level 2
        kmeans3, labels3 = multi_level_clustering(cluster_data2, 5)
        cluster_hierarchy[(1, i, j)] = (kmeans3, labels3)
        
        for k in range(5):
            cluster_data3 = cluster_data2[labels3 == k]
            cluster_coords3 = cluster_coords2[labels3 == k]
            if len(cluster_data3) >=5:
            
                # Level 4: 5 Sub-Clusters within each of the 5 sub-clusters from Level 3
                kmeans4, labels4 = multi_level_clustering(cluster_data3, 5)
                cluster_hierarchy[(1, i, j, k)] = (kmeans4, labels4)
            else:
                cluster_hierarchy[(1, i, j, k)] = (None, None)
            
            for l in range(5):
                sub_cluster_data = cluster_data3[labels4 == l]
                sub_cluster_coords = cluster_coords3[labels4 == l]
                
                # Train a regression model for each sub-cluster
                model = Decision()
                model.fit(sub_cluster_data, sub_cluster_coords)
                
                # Store the model
                model_hierarchy[(1, i, j, k, l)] = model

# Step 2: Evaluation Function
def predict_coordinates(embedding, cluster_hierarchy, model_hierarchy):
    """Predict the coordinates for a given image embedding using the multi-level clustering."""
    # Level 1 Prediction
    kmeans1, _ = cluster_hierarchy[1]
    label1 = kmeans1.predict([embedding])[0]
    
    # Level 2 Prediction
    kmeans2, _ = cluster_hierarchy[(1, label1)]
    label2 = kmeans2.predict([embedding])[0]
    
    # Level 3 Prediction
    kmeans3, _ = cluster_hierarchy[(1, label1, label2)]
    label3 = kmeans3.predict([embedding])[0]
    
    # Level 4 Prediction
    kmeans4, _ = cluster_hierarchy[(1, label1, label2, label3)]
    label4 = kmeans4.predict([embedding])[0]
    
    # Final Prediction using the trained model
    model = model_hierarchy[(1, label1, label2, label3, label4)]
    predicted_coords = model.predict([embedding])[0]
    
    return predicted_coords

def evaluate_model(X_test, y_test, cluster_hierarchy, model_hierarchy):
    """Evaluate the overall model on a test set."""
    print("Evaluating the model...")
    y_pred = np.array([predict_coordinates(embedding, cluster_hierarchy, model_hierarchy) for embedding in track(X_test, description="Predicting coordinates...")])
    mse = np.sqrt(mean_squared_error(y_test, y_pred)) * 111.1111  # Convert to kilometers
    
    # Calculate the distance errors with wraparound on the globe
    distances = []
    for i in range(len(y_test)):
        lat_true, lon_true = y_test[i]
        lat_pred, lon_pred = y_pred[i]
        distance = np.sqrt((lat_true - lat_pred)**2 + (lon_true - lon_pred)**2)
        distances.append(min(distance, 360 - distance))
    
    distances = np.array(distances) * 111.1111  # Convert to kilometers
    
    # Generate a histogram of the distance errors
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black', linewidth=1.2)
    plt.title('Histogram of Distance Errors')
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.show()
    
    return mse

# Step 3: Evaluate the Model
mse = evaluate_model(X_test, y_test, cluster_hierarchy, model_hierarchy)
print(f"Mean Squared Error: {mse} km")

# Visualization: True vs Predicted Coordinates
plt.figure(figsize=(10, 8))

# World map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')

# Plot true coordinates
plt.scatter(y_test[:, 1], y_test[:, 0], color='blue', label='True')

# Plot predicted coordinates
y_pred = np.array([predict_coordinates(embedding, cluster_hierarchy, model_hierarchy) for embedding in track(X_test, description="Visualizing predictions...")])
plt.scatter(y_pred[:, 1], y_pred[:, 0], color='red', label='Predicted')

# Draw lines between true and predicted coordinates
for i in range(len(y_test)):
    plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='gray', alpha=0.5)

plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
