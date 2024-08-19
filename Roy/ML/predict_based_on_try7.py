import os
import torch
import numpy as np
import joblib
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
import warnings

warnings.filterwarnings("ignore")

# Function to load models
def load_models():
    dt_classifier = joblib.load('Roy/ML/Saved_Models/random_forest_classifier.pkl')
    sub_clusters = {}
    sub_cluster_models = {}

    for continent in range(7):  # Assuming 7 continents
        sub_kmeans = joblib.load(f'Roy/ML/Saved_Models/kmeans_continent_{continent}.pkl')
        sub_clusters[continent] = sub_kmeans

        for sub_label in range(25):  # Assuming 25 sub-clusters per continent
            try:
                kr_regressor = joblib.load(f'Roy/ML/Saved_Models/kernel_ridge_continent_{continent}_subcluster_{sub_label}.pkl')
                sub_cluster_models[(continent, sub_label)] = kr_regressor
            except FileNotFoundError:
                continue

    return dt_classifier, sub_clusters, sub_cluster_models

# Custom Model
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super(GeoEmbeddingModel, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final classification layer
        #self.fc = nn.Linear(2048, 4096)  # Add a fully connected layer

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x

# Load the models
dt_classifier, sub_clusters, sub_cluster_models = load_models()

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict coordinates from an image file
def predict_from_image(image_path, model, dt_classifier, sub_cluster_models, sub_clusters):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    # Generate the embedding
    embedding = model(image).detach().cpu().numpy().flatten()

    # Predict the coordinates
    dt_classifier.n_jobs = 1
    dt_classifier.verbose = 0
    continent = dt_classifier.predict([embedding])[0]
    
    sub_kmeans = sub_clusters[continent]
    sub_label = sub_kmeans.predict([embedding])[0]

    predicted_coords = sub_cluster_models[(continent, sub_label)].predict([embedding])[0]

    return predicted_coords

# Load the pretrained embedding model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GeoEmbeddingModel().to(device)
model.eval()  # Set the model to evaluation mode
model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model.pth'))

# Example Usage
image_folder = "Roy/Test_Images"

# Select a random image from the folder
image_path = os.path.join(image_folder, np.random.choice(os.listdir(image_folder)))
predicted_coords = predict_from_image(image_path, model, dt_classifier, sub_cluster_models, sub_clusters)
print(f"Predicted Coordinates: {predicted_coords}")

# place the predicted coordinates on the map
import geopandas as gpd
import matplotlib.pyplot as plt



# open google maps with the predicted coordinates
import webbrowser

url = f"https://www.google.ch/maps/@{predicted_coords[0]},{predicted_coords[1]},9z?entry=ttu"

# Open URL in a new tab, if a browser window is already open.
webbrowser.open_new_tab(url)

#open the image
from PIL import Image
image = Image.open(image_path)
image.show()





print(f"Image: {image_path}")

# Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a GeoDataFrame with the predicted coordinates
df = gpd.GeoDataFrame({'Latitude': [predicted_coords[0]], 'Longitude': [predicted_coords[1]]},
                      geometry=gpd.points_from_xy([predicted_coords[1]], [predicted_coords[0]]))
                      
# Plot the world map
fig, ax = plt.subplots(figsize=(12, 8))
world.boundary.plot(ax=ax, linewidth=1)
df.plot(ax=ax, markersize=50, color='red')
plt.title('Predicted Coordinates on World Map')
plt.show()