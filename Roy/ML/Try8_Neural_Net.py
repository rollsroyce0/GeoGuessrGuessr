import os
import torch
import numpy as np # Import joblib for saving and loading models
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
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

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x

# Load or generate embeddings
if os.path.exists('Roy/ML/embeddings.npy'):
    embeddings = np.load('Roy/ML/embeddings.npy').astype(np.float32)  # Ensure embeddings are float64
    image_paths = np.load('Roy/ML/image_paths.npy')
else:
    # Load image paths and extract coordinates
    image_paths = [os.path.join(location, img_file) for img_file in os.listdir(location)]
    coordinates = np.array([extract_coordinates(path) for path in image_paths])

    # Initialize the custom model
    model = GeoEmbeddingModel().to(device)
    
    # Dataset and DataLoader
    dataset = GeoDataset(image_paths=image_paths, coordinates=coordinates, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Load the pre-trained model if available
    if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model.pth'):
        model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model.pth'))

    # Generate embeddings
    print("Generating custom embeddings...")
    embeddings = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, _ in track(dataloader, description="Processing images..."):
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy().astype(np.float32))  # Move to CPU and convert to float64
    embeddings = np.vstack(embeddings)

    # Save the embeddings and image paths
    np.save('Roy/ML/embeddings.npy', embeddings)
    np.save('Roy/ML/image_paths.npy', np.array(image_paths))
    
    # save the model
    torch.save(model.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")

# generate a pareto chart of the embeddings
plt.figure(figsize=(10, 6))
plt.scatter(embeddings[:, 0], embeddings[:, 1], color='skyblue', alpha=0.5)
plt.title('Embeddings Pareto Chart')
plt.xlabel('Embedding Dimension 1')
plt.ylabel('Embedding Dimension 2')
plt.show()

# Extract the latitude and longitude from each image path
coordinates = np.array([extract_coordinates(path) for path in image_paths])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, coordinates, test_size=5000/embeddings.shape[0])

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)  # Fully connected layer
        self.dropout0 = nn.Dropout(0.05)   # Dropout layer to prevent overfitting
        self.batch_norm1 = nn.BatchNorm1d(1024)  # Batch normalization layer
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.05)   # Dropout layer to prevent overfitting

        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(512, 1024)
        self.batch_norm3 = nn.BatchNorm1d(1024)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(1024, 256)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(0.1)
        
        self.fc5 = nn.Linear(256, 64)
        self.batch_norm5 = nn.BatchNorm1d(64)
        self.gelu5 = nn.GELU()
        self.dropout5 = nn.Dropout(0.1)
        
        self.fc6 = nn.Linear(64, 32)
        self.batch_norm6 = nn.BatchNorm1d(32)
        self.gelu6 = nn.GELU()
        self.dropout6 = nn.Dropout(0.1)
        

        self.fc7 = nn.Linear(32, 2)


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

# Initialize the model
geo_predictor = GeoPredictorNN().to(device)

def degrees_to_radians(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


# Haversine distance calculation
def haversine_loss(coords1, coords2):
    lat1, lon1 = coords1[:, 0], coords1[:, 1]
    lat2, lon2 = coords2[:, 0], coords2[:, 1]

    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))

    distance = 6371.01 * c  # Earth's radius is approximately 6371.01 km
    return distance.mean()

# Define the loss function and optimizer
criterion = haversine_loss
optimizer = optim.AdamW(geo_predictor.parameters())

# Prepare DataLoader for training
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)

# Training loop
epochs = 20 # You can adjust the number of epochs
losses = []
val_losses = []
for epoch in track(range(epochs), description="Training the model..."):
    geo_predictor.train()
    running_loss = 0.0
    for batch in train_loader:
        embeddings_batch, coords_batch = batch
        embeddings_batch, coords_batch = embeddings_batch.float().to(device), coords_batch.float().to(device)

        optimizer.zero_grad()
        outputs = geo_predictor(embeddings_batch)
        loss = criterion(outputs, coords_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    losses.append(running_loss/len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Calculate the validation loss
    geo_predictor.eval()
    with torch.no_grad():
        val_loss = haversine_loss(geo_predictor(torch.tensor(X_test).float().to(device)), torch.tensor(y_test).float().to(device))
        print(f"Validation Loss: {val_loss.item():.4f}")
    
    val_losses.append(val_loss.item())
    
    # Save the model checkpoint every 10 epochs
    if (epoch+1) % 3 == 0:
        torch.save(geo_predictor.state_dict(), f'Roy/ML/Saved_Models/Checkpoint_Models_NN/geo_predictor_nn_{epoch}_loss_{np.round(val_loss.item(), 0)}.pth')
        
    # if after 50 epochs the validation loss is above 1650, reset the model
    if (epoch+1) % 100 == 0 and val_loss.item() > 1350:
        geo_predictor = GeoPredictorNN().to(device)
        optimizer = optim.AdamW(geo_predictor.parameters())
        print("Resetting the model...")
    
    

# Save the trained model
torch.save(geo_predictor.state_dict(), 'Roy/ML/Saved_Models/geo_predictor_nn.pth')

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, color='skyblue')
plt.plot(val_losses, color='orange')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

def haversine_distance(coords1, coords2):
    return geopy.distance.geodesic(coords1, coords2).kilometers


# Step 3: Evaluation Function
def predict_coordinates_nn(embedding, geo_predictor):
    """Predict coordinates using the trained neural network model."""
    geo_predictor.eval()
    with torch.no_grad():
        embedding = torch.tensor(embedding).float().unsqueeze(0).to(device)
        predicted_coords = geo_predictor(embedding).cpu().numpy()[0]
        
        # Convert the predicted coordinates to the correct range
        predicted_coords[0] = (predicted_coords[0] + 90) % 180 - 90
        predicted_coords[1] = (predicted_coords[1] + 180) % 360 - 180
        
    return predicted_coords

def evaluate_nn_model(X_test, y_test, geo_predictor):
    """Evaluate the neural network model using Haversine distance."""
    print("Evaluating the neural network model...")
    y_pred = np.array([predict_coordinates_nn(embedding, geo_predictor) for embedding in track(X_test, description="Predicting coordinates...")])

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
    plt.title('Histogram of Distance Errors (NN)')
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.show()

    return np.mean(distances)

# Load the trained neural network model
geo_predictor.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_predictor_nn.pth'))

# Evaluate the model
mean_haversine_distance_nn = evaluate_nn_model(X_test, y_test, geo_predictor)
print(f"Mean Haversine Distance with NN: {mean_haversine_distance_nn} km")

# Visualization: True vs Predicted Coordinates
plt.figure(figsize=(10, 8))

# World map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.boundary.plot(ax=plt.gca(), linewidth=1, color='black')

# Plot true coordinates
plt.scatter(y_test[:100, 1], y_test[:100, 0], color='blue', label='True')

# Plot predicted coordinates
y_pred = np.array([predict_coordinates_nn(embedding, geo_predictor) for embedding in X_test[:100]])
plt.scatter(y_pred[:100, 1], y_pred[:100, 0], color='red', label='Predicted')

# Draw lines between true and predicted coordinates
for i in range(100):
    plt.plot([y_test[i, 1], y_pred[i, 1]], [y_test[i, 0], y_pred[i, 0]], color='green', alpha=0.5)

plt.title('True vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()