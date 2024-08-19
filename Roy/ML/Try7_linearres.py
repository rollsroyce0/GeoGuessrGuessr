import os
import torch
import numpy as np
import joblib  # Import joblib for saving and loading models
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
import geopy.distance
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import geopandas as gpd
import warnings

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

# Custom Dataset with Image Variations
class GeoDataset(Dataset):
    def __init__(self, image_paths, coordinates, transform=None):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def apply_variations(self, image):
        variations = []

        # Original image
        variations.append(image)

        # Downsize image
        downsized_image = image.resize((image.size[0] // 2, image.size[1] // 2), Image.ANTIALIAS)
        variations.append(downsized_image)

        # Zoom-in (crop center and resize)
        width, height = image.size
        zoomed_image = image.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4))
        zoomed_image = zoomed_image.resize((width, height), Image.ANTIALIAS)
        variations.append(zoomed_image)

        # Mirrored image
        mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        variations.append(mirrored_image)

        # Grayscale image
        grayscale_image = image.convert('L').convert('RGB')  # Convert to grayscale then back to RGB
        variations.append(grayscale_image)

        # Warped image (affine transformation)
        warp_matrix = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
        warped_image = image.transform(image.size, Image.AFFINE, warp_matrix.flatten()[:6])
        variations.append(warped_image)

        return variations

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        coords = self.coordinates[idx]

        # Apply variations
        images = self.apply_variations(image)
        
        # Apply transformations
        if self.transform:
            images = [self.transform(img) for img in images]

        return images, coords

# Load or generate embeddings with variations
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
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Reduce batch size due to increased computation

    # Load the pre-trained model if available
    if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model.pth'):
        model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model.pth'))

    # Generate embeddings
    print("Generating custom embeddings with variations...")
    embeddings = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, _ in track(dataloader, description="Processing images..."):
            images = torch.cat(images, 0).to(device)  # Concatenate the images (batch_size * num_variations)
            output = model(images)
            embeddings.append(output.cpu().numpy().astype(np.float64))  # Move to CPU and convert to float64
    embeddings = np.vstack(embeddings)

    # Save the embeddings and image paths
    np.save('Roy/ML/embeddings.npy', embeddings)
    np.save('Roy/ML/image_paths.npy', np.array(image_paths))
    
    # Save the model
    torch.save(model.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")


# Extract the latitude and longitude from each image path
coordinates = np.array([extract_coordinates(path) for path in image_paths])

# Function to assign continent labels based on coordinates
def assign_continent(lat, lon):
    if lat < 30 and -30 <= lon <= 60:
        return 0  # Africa
    elif lat > -13 and lon > 45:
        return 1  # Asia
    elif -50 < lat < -13 and 110 <= lon <= 180:
        return 2  # Australia
    elif lat > 12 and -130 <= lon <= -30:
        return 3  # North America
    elif lat < 12 and -90 <= lon <= -30:
        return 4  # South America
    elif lat > 30 and -30 <= lon <= 45:
        return 5  # Europe
    else:
        return 6  # Others (e.g., Pacific islands, Antarctica)

# Assign continent labels to coordinates
continent_labels = np.array([assign_continent(lat, lon) for lat, lon in coordinates])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, train_labels, test_labels = train_test_split(embeddings, coordinates, continent_labels, test_size=5000/embeddings.shape[0])

# Step 1: Train a Classifier to Predict Continents
print("Training RandomForestClassifier on continent labels...")
dt_classifier = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
dt_classifier.fit(X_train, train_labels)

# print the accuracy
print(f"Training Accuracy: {dt_classifier.score(X_train, train_labels)}")

# Save the trained RandomForestClassifier
joblib.dump(dt_classifier, 'Roy/ML/Saved_Models/random_forest_classifier.pkl')

# Function to train sub-cluster models using Kernelized Lasso-like regression
def train_sub_cluster_model(sub_cluster_data, sub_cluster_coordinates):
    # Kernelized Lasso-like regressor using Lasso
    kr_regressor = Lasso(alpha=0.001)
    kr_regressor.fit(sub_cluster_data, sub_cluster_coordinates)
    return kr_regressor

# Step 2: Sub-Clustering and Training Regressors
sub_clusters = {}
sub_cluster_models = {}

print("Sub-clustering within each continent and training models...")
for continent in track(np.unique(train_labels), description="Processing continents..."):
    print(f"Processing continent {continent} with {np.sum(train_labels == continent)} samples...")
    # Filter the data for this continent
    continent_data = X_train[train_labels == continent]
    continent_coordinates = y_train[train_labels == continent]

    # Sub-cluster using KMeans
    cluster_n = 6*int(np.sqrt(np.sqrt(len(continent_data))))
    print(f"Sub-clustering into {cluster_n} sub-clusters...")
    sub_kmeans = KMeans(n_clusters=cluster_n)
    sub_labels = sub_kmeans.fit_predict(continent_data)

    # Store the sub-cluster information
    sub_clusters[continent] = sub_kmeans

    # Save the KMeans model for each continent
    joblib.dump(sub_kmeans, f'Roy/ML/Saved_Models/kmeans_continent_{continent}.pkl')

    # Train a regression model for each sub-cluster
    for sub_label in range(cluster_n):
        sub_cluster_data = continent_data[sub_labels == sub_label]
        sub_cluster_coordinates = continent_coordinates[sub_labels == sub_label]

        kr_regressor = train_sub_cluster_model(sub_cluster_data, sub_cluster_coordinates)
        
        # Store the model for later prediction
        sub_cluster_models[(continent, sub_label)] = kr_regressor

        # Save the trained KernelRidge model for each sub-cluster
        joblib.dump(kr_regressor, f'Roy/ML/Saved_Models/kernel_ridge_continent_{continent}_subcluster_{sub_label}.pkl')

print("All models saved successfully.")


# Haversine distance calculation
def haversine_distance(coords1, coords2):
    return geopy.distance.geodesic(coords1, coords2).kilometers

# Step 3: Evaluation Function
def predict_coordinates(embedding, dt_classifier, sub_cluster_models, sub_clusters):
    """Predict the coordinates for a given image embedding."""
    # Predict the continent
    dt_classifier.n_jobs = 1  # Workaround for pickling the classifier with n_jobs=-1
    dt_classifier.verbose = 0
    continent = dt_classifier.predict([embedding])[0]

    # Predict the sub-cluster within the continent
    sub_kmeans = sub_clusters[continent]
    sub_label = sub_kmeans.predict([embedding])[0]

    # Final prediction using the regression model
    predicted_coords = sub_cluster_models[(continent, sub_label)].predict([embedding])[0]

    return predicted_coords

def evaluate_model(X_test, y_test, dt_classifier, sub_cluster_models, sub_clusters):
    """Evaluate the overall model on a test set using Haversine distance."""
    print("Evaluating the model...")
    y_pred = np.array([predict_coordinates(embedding, dt_classifier, sub_cluster_models, sub_clusters) for embedding in track(X_test, description="Predicting coordinates...")])

    # Calculate Haversine distances
    distances = np.array([haversine_distance(y_test[i], y_pred[i]) for i in range(len(y_test))])

    # Calculate and display statistics
    print(f"Mean Distance Error: {np.mean(distances)} km")
    print(f"Median Distance Error: {np.median(distances)} km")
    print(f"Max Distance Error: {np.max(distances)} km")
    print(f"Min Distance Error: {np.min(distances)} km")
    print(f"Standard Deviation: {np.std(distances)} km")
    print(f"25th Percentile: {np.percentile(distances, 25)} km")
    print(f"50th Percentile: {np.percentile(distances, 50)} km")
    print(f"75th Percentile: {np.percentile(distances, 75)} km")
    print(f"90th Percentile: {np.percentile(distances, 90)} km")
    print(f"Index of the minimum distance: {np.argmin(distances)} with name {image_paths[np.argmin(distances)]}")
    
    # Generate a histogram of the distance errors
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black', linewidth=1.2)
    plt.title('Histogram of Distance Errors')
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.show()

    return np.mean(distances)

# Step 4: Evaluate the Model with Haversine Distance
mean_haversine_distance = evaluate_model(X_test, y_test, dt_classifier, sub_cluster_models, sub_clusters)
print(f"Mean Haversine Distance: {mean_haversine_distance} km")

# Visualization: True vs Predicted Coordinates
plt.figure(figsize=(10, 8))

# World map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')

# Plot true coordinates
plt.scatter(y_test[:100, 1], y_test[:100, 0], color='blue', label='True')

# Plot predicted coordinates
y_pred = np.array([predict_coordinates(embedding, dt_classifier, sub_cluster_models, sub_clusters) for embedding in track(X_test, description="Visualizing predictions...")])
plt.scatter(y_pred[:100, 1], y_pred[:100, 0], color='red', label='Predicted')

# Draw lines between true and predicted coordinates
for i in range(len(y_test)):
    if i%100 == 99:
        plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='gray', alpha=0.5)

plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
