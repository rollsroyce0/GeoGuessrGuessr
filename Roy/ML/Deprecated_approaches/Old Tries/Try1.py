import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if os.path.exists('Roy/ML/embeddings.npy'):
    embeddings = np.load('Roy/ML/embeddings.npy')
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
        transforms.Resize((480, 480)),
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
        return embedding.squeeze().cpu().numpy()  # Convert to numpy and remove batch dimension, and move back to CPU

    # List to store embeddings and image paths
    embeddings = []
    image_paths = []

    # Iterate through the images and generate embeddings
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
    np.save('Roy/ML/embeddings.npy', embeddings)
    np.save('Roy/ML/image_paths.npy', image_paths)

# Convert embeddings list to numpy array
embeddings = np.array(embeddings)

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")

# Step 2: Cluster Embeddings into 10 groups
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Save the labels with their corresponding image paths
image_classification_df = pd.DataFrame({
    'image_path': image_paths,
    'cluster_label': labels
})

# Save to a CSV file for later reference
image_classification_df.to_csv('image_classification.csv', index=False)

# Display the number of images per cluster
print(image_classification_df['cluster_label'].value_counts())

# Step 3: PCA to reduce the dimensionality of the embeddings to 2D and visualize the clusters
print("PCA")
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Visualize the clusters
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.title('Image Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Step 4: Display Images from Each Cluster
def display_images_from_cluster(cluster_label, num_images=5):
    cluster_images = image_classification_df[image_classification_df['cluster_label'] == cluster_label]['image_path']
    num_images = min(num_images, len(cluster_images))

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Cluster {cluster_label}")

    for i in range(num_images):
        img_path = cluster_images.iloc[i]
        img = Image.open(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.show()

# Now check that images with the same location are in the same cluster, the location is in the image name (before the second underscore)
def extract_location(image_path):
    res = image_path.split('_')[0] + ',' + image_path.split('_')[1]
    res = res.replace('D:/GeoGuessrGuessr/geoguesst', '')
    res = res.replace('\\', '')
    return res

# Extract the location from each image path
image_classification_df['location'] = image_classification_df['image_path'].apply(extract_location)

# plot the locations on a world map with different colors for each cluster
import geopandas as gpd
from shapely.geometry import Point

# Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a GeoDataFrame from the image locations
image_locations = image_classification_df['location'].str.split(',', expand=True)
image_locations.columns = ['latitude', 'longitude']
image_locations = image_locations.astype(float)
image_locations['cluster_label'] = image_classification_df['cluster_label']
image_locations['cluster_label'] = image_locations['cluster_label'].astype('category')
image_locations['geometry'] = image_locations.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
image_locations = gpd.GeoDataFrame(image_locations, geometry='geometry')

# Plot the world map
fig, ax = plt.subplots(figsize=(15, 10))
world.boundary.plot(ax=ax)

# Plot the image locations with colors based on the cluster_label
image_locations.plot(ax=ax, 
                     markersize=10, 
                     column='cluster_label', 
                     cmap='tab10', 
                     legend=True, 
                     categorical=True)

# Update legend title
legend = ax.get_legend()
legend.set_title('Cluster Label')

# Set plot title
plt.title('Image Clustering by Location')
plt.show()
