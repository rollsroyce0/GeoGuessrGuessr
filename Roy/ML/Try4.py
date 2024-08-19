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
import geopy.distance

warnings.filterwarnings("ignore")
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

init_size = 100
# Step 1: Initial Clustering into 50 Clusters
print("Clustering into init_size initial clusters...")
kmeans_initial = KMeans(n_clusters=init_size, random_state=42)
initial_labels = kmeans_initial.fit_predict(X_train)

# Step 2: Sub-Clustering with Elbow Criterion
def find_optimal_clusters(max_k=30):
    """Use the elbow method to find the optimal number of clusters."""
    iters = np.random.randint(5, max_k)
    return iters

# Dictionary to hold sub-clusters and models
sub_clusters = {}
sub_cluster_models = {}

print("Sub-clustering within each initial cluster and training models...")
for cluster_label in track(range(init_size), description="Processing clusters..."):
    # Filter the data for this cluster
    cluster_data = X_train[initial_labels == cluster_label]
    cluster_coordinates = y_train[initial_labels == cluster_label]
    
    # Determine the optimal number of sub-clusters using the elbow criterion
    optimal_k = find_optimal_clusters()
    print(f"Cluster {cluster_label}: Selected k = {optimal_k}")

    # Perform the sub-clustering
    sub_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    sub_labels = sub_kmeans.fit_predict(cluster_data)
    
    # Store the sub-cluster information
    sub_clusters[cluster_label] = sub_kmeans
    
    # Train a model for each sub-cluster
    for sub_label in range(optimal_k):
        sub_cluster_data = cluster_data[sub_labels == sub_label]
        sub_cluster_coordinates = cluster_coordinates[sub_labels == sub_label]
        
        # Train a small regression model (e.g., DecisionTreeRegressor)
        model = Decision()
        model.fit(sub_cluster_data, sub_cluster_coordinates)
        
        # Store the model for later prediction
        sub_cluster_models[(cluster_label, sub_label)] = model

# Step 3: Evaluation Function
def predict_coordinates(embedding, initial_kmeans, sub_cluster_models, sub_clusters):
    """Predict the coordinates for a given image embedding."""
    # Predict the initial cluster
    initial_label = initial_kmeans.predict([embedding])[0]
    
    # Predict the sub-cluster
    sub_kmeans = sub_clusters[initial_label]
    sub_label = sub_kmeans.predict([embedding])[0]
    
    # Predict the coordinates using the model trained on this sub-cluster
    model = sub_cluster_models[(initial_label, sub_label)]
    predicted_coords = model.predict([embedding])[0]
    
    return predicted_coords

def evaluate_model(X_test, y_test, initial_kmeans, sub_cluster_models, sub_clusters):
    """Evaluate the overall model on a test set."""
    print("Evaluating the model...")
    y_pred = np.array([predict_coordinates(embedding, initial_kmeans, sub_cluster_models, sub_clusters) for embedding in track(X_test, description="Predicting coordinates...")])
    mse = np.sqrt(mean_squared_error(y_test, y_pred))*111.1111  # Convert to kilometers
    
    # Calculate the distance errors with wraparound on the globe
    distances = []
    for i in range(len(y_test)):
        lat_true, lon_true = y_test[i]
        lat_pred, lon_pred = y_pred[i]
        # use geopy.distance to calculate distance between two points on the globe
        distance = geopy.distance.distance((lat_true, lon_true), (lat_pred, lon_pred)).km
        distances.append(distance)
    
    # Print statistics
    print(f"Mean Distance Error: {np.mean(distances)} km")
    print(f"Median Distance Error: {np.median(distances)} km")
    print(f"Max Distance Error: {np.max(distances)} km")
    print(f"Min Distance Error: {np.min(distances)} km")
    # generate a histogram of the distance errors
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black', linewidth=1.2)
    plt.title('Histogram of Distance Errors')
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.show()
    
    
    return mse

# Step 4: Evaluate the Model
mse = evaluate_model(X_test, y_test, kmeans_initial, sub_cluster_models, sub_clusters)
print(f"Mean Squared Error: {mse} km")

# Visualization: True vs Predicted Coordinates
plt.figure(figsize=(10, 8))

# World map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')

# Plot true coordinates
plt.scatter(y_test[:, 1], y_test[:, 0], color='blue', label='True')

# Plot predicted coordinates
y_pred = np.array([predict_coordinates(embedding, kmeans_initial, sub_cluster_models, sub_clusters) for embedding in track(X_test, description="Visualizing predictions...")])
plt.scatter(y_pred[:, 1], y_pred[:, 0], color='red', label='Predicted')

# Draw lines between true and predicted coordinates
for i in range(len(y_test)):
    plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='gray', alpha=0.5)

plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
