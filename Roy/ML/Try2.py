import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as Decision
from sklearn.metrics import mean_squared_error
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


# perform PCA to reduce the dimensionality of the embeddings and generate a paretto plot to visualize the explained variance

# pareto plot
def pareto_plot(explained_variance, title='Pareto Plot', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(range(len(explained_variance)), explained_variance, color='b')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_xlabel('Principal Components')
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(range(len(explained_variance)), np.cumsum(explained_variance), color='r', marker='D', ms=7)
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.grid(False)
    plt.show()
    
# Perform PCA to reduce the dimensionality of the embeddings
#from sklearn.decomposition import PCA

#pca = PCA(n_components=0.9)  # Retain 90% of the variance
#embeddings_pca = pca.fit_transform(embeddings)

#embeddings = embeddings_pca

# Visualize the explained variance
#pareto_plot(pca.explained_variance_ratio_ * 100, title='Pareto Plot of Explained Variance')

#quit()

# Extract the latitude and longitude from each image path
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

coordinates = np.array([extract_coordinates(path) for path in image_paths])




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, coordinates, test_size=0.2, random_state=42)

# Train a regression model (e.g., RandomForestRegressor)
model = Decision()
print("Training the model...")
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = np.sqrt(mean_squared_error(y_test, y_pred))*111.1111  # Convert to kilometers
print(f"Squared Error: {mse}")


import geopandas as gpd
from shapely.geometry import Point

# Visualize the true vs. predicted coordinates. Include a world map for reference
plt.figure(figsize=(10, 8))

# World map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')


plt.scatter(y_test[0, 1], y_test[0, 0], color='blue', label='True')
plt.scatter(y_pred[0, 1], y_pred[0, 0], color='red', label='Predicted')
plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
