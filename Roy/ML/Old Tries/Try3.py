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

warnings.filterwarnings("ignore")

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load or generate embeddings
if os.path.exists('Roy/ML/embeddings.npy'):
    embeddings = np.load('Roy/ML/embeddings.npy').astype(np.float64)  # Ensure embeddings are float64
    image_paths = np.load('Roy/ML/image_paths.npy')
else:
    # Define the location of the images
    location = "D:/GeoGuessrGuessr/geoguesst"

    # Load the pre-trained ResNet50 model and move it to the GPU if available
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
    model = model.to(device)  # Move model to GPU if available
    model.eval()  # Set the model to evaluation mode

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to load and preprocess images
    def load_and_preprocess_image(img_path):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)  # Add a batch dimension
        img = img.to(device)  # Move tensor to GPU if available
        return img

    # Function to generate embeddings using the model
    def generate_embeddings(model, img_tensor):
        with torch.no_grad():  # Disable gradient computation
            embedding = model(img_tensor)
        return embedding.squeeze().cpu().numpy().astype(np.float64)  # Convert to numpy (float64) and remove batch dimension, and move back to CPU

    # List to store embeddings and image paths
    embeddings = []
    image_paths = []

    # Iterate through the images and generate embeddings
    print("Processing images and generating embeddings...")
    for img_file in track(os.listdir(location), description="Processing images..."):
        img_path = os.path.join(location, img_file)
        try:
            img_tensor = load_and_preprocess_image(img_path)
            embedding = generate_embeddings(model, img_tensor)
            embeddings.append(embedding.flatten())  # Flatten the output to a 1D vector
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # Save the embeddings
    np.save('Roy/ML/embeddings.npy', np.array(embeddings))
    np.save('Roy/ML/image_paths.npy', np.array(image_paths))

# Convert embeddings list to numpy array
embeddings = np.array(embeddings)

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")

# Extract the latitude and longitude from each image path
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

coordinates = np.array([extract_coordinates(path) for path in image_paths])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, coordinates, test_size=5/embeddings.shape[0])

# Step 1: Initial Clustering into 25 Clusters
print("Clustering into 25 initial clusters...")
kmeans_initial = KMeans(n_clusters=50)
initial_labels = kmeans_initial.fit_predict(X_train)

# Step 2: Sub-Clustering with Elbow Criterion
def find_optimal_clusters(max_k=50):
    """Use the elbow method to find the optimal number of clusters."""
    iters = np.random.randint(15, max_k)
    return iters

# Dictionary to hold sub-clusters and models
sub_clusters = {}
sub_cluster_models = {}

print("Sub-clustering within each initial cluster and training models...")
for cluster_label in track(range(50), description="Processing clusters..."):
    # Filter the data for this cluster
    cluster_data = X_train[initial_labels == cluster_label]
    cluster_coordinates = y_train[initial_labels == cluster_label]
    
    # Determine the optimal number of sub-clusters using the elbow criterion
    iters = find_optimal_clusters()
    optimal_k = iters # Choose the optimal k
    print(f"Cluster {cluster_label}: Selected k = {optimal_k}")

    # Perform the sub-clustering
    sub_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    sub_labels = sub_kmeans.fit_predict(cluster_data)
    
    # Store the sub-cluster information
    sub_clusters[cluster_label] = sub_kmeans
    
    # Train a model for each sub-cluster
    for sub_label in range(optimal_k):
        sub_cluster_data = cluster_data[sub_labels == sub_label]
        #print("Number of images in sub-cluster:", len(sub_cluster_data))
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
    mse = np.sqrt(mean_squared_error(y_test, y_pred)) * 111.1111  # Convert to kilometers
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

# also draw lines between true and predicted coordinates
for i in range(len(y_test)):
    plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='gray', alpha=0.5)

plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
