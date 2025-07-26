import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import geopandas as gpd
import matplotlib.pyplot as plt
import webbrowser
import math
import warnings

warnings.filterwarnings("ignore")

# Custom Model to generate embeddings
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super(GeoEmbeddingModel, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final classification layer

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x

# Custom model for predicting coordinates
class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)  # Fully connected layer
        self.dropout0 = nn.Dropout(0.2)   # Dropout layer to prevent overfitting
        self.batch_norm1 = nn.BatchNorm1d(1024)  # Batch normalization layer
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)   # Dropout layer to prevent overfitting

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

# Function to load and transform the image
def load_and_transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Function to predict the coordinates of the image
def predict_image_coordinates(image_path, geo_embedding_model, geo_predictor_nn):
    # Load and transform the image
    image_tensor = load_and_transform_image(image_path).to(device)
    
    # Generate the embedding
    geo_embedding_model.eval()
    with torch.no_grad():
        embedding = geo_embedding_model(image_tensor).cpu().numpy().squeeze()
    
    # Predict coordinates
    geo_predictor_nn.eval()
    with torch.no_grad():
        embedding_tensor = torch.tensor(embedding).float().unsqueeze(0).to(device)
        predicted_coords = geo_predictor_nn(embedding_tensor).cpu().numpy()[0]
        
        # Convert the predicted coordinates to the correct range
        predicted_coords[0] = (predicted_coords[0] + 90) % 180 - 90
        predicted_coords[1] = (predicted_coords[1] + 180) % 360 - 180
    
    return predicted_coords

def geoguessr_points_formula(error):
    # Convert the error to GeoGuessr points, not perfect, but very close
    
    if error < 0.15:
        return 5000
    else:
        return np.floor(5000 * math.e**(-1*error/2000))

def haversine(coord1, coord2):
    """
    Calculate the great circle distance in kilometers between two points on the Earth specified in decimal degrees.
    """
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# Visualization: Plot predicted coordinates on the world map
def plot_coordinates_on_map(predicted_coords, image_path):
    plt.figure(figsize=(10, 8))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')

    plt.scatter(predicted_coords[1], predicted_coords[0], color='red', label='Predicted Location', s=100)
    plt.title(f'Predicted Coordinates on the World Map for the Image {image_path}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the models
    geo_embedding_model = GeoEmbeddingModel().to(device)
    geo_predictor_nn = GeoPredictorNN().to(device)

    # Load the saved model weights
    geo_embedding_model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth', map_location=device))
    geo_predictor_nn.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_predictor_nn_500e_64b_952k.pth', map_location=device))
    # currently best model: geo_predictor_nn_500e_64b_952k.pth at 16037 points
    
    # Call the images to be tested Current_Test_ImagesX.jpg

    counter = 0
    errors = []
    for image_path in os.listdir('Roy/Test_Images'):
        
        if not image_path.endswith('.jpg'):
            continue
        if not image_path.__contains__("Current"):
            continue    
        
        image_path = f"Roy/Test_Images/{image_path}"
        
        #image_path = "Roy/Test_Images/test10.png"
        
        # Predict the coordinates
        predicted_coords = predict_image_coordinates(image_path, geo_embedding_model, geo_predictor_nn)
        print(f"Predicted Coordinates: Latitude: {predicted_coords[0]}, Longitude: {predicted_coords[1]}")
        

        counter += 1
        # Open the location in Google Maps
        url = f"https://www.google.com/maps/@{predicted_coords[0]},{predicted_coords[1]},9z"
        webbrowser.open_new_tab(url)
        
        # Plot the predicted coordinates on the world map
        plot_coordinates_on_map(predicted_coords, image_path)

    

    
