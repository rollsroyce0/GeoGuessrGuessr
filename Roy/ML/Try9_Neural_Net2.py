import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import geopy.distance
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import geopandas as gpd
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        self.backbone = models.resnet50(pretrained=True)  # Use ResNet-50 for faster performance
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final classification layer

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x

# Load or generate embeddings
if os.path.exists('Roy/ML/embeddings.npy'):
    embeddings = np.load('Roy/ML/embeddings.npy').astype(np.float32)  # Ensure embeddings are float32
    image_paths = np.load('Roy/ML/image_paths.npy')
else:
    # Load image paths and extract coordinates
    image_paths = [os.path.join(location, img_file) for img_file in os.listdir(location)]
    coordinates = np.array([extract_coordinates(path) for path in image_paths])

    # Initialize the custom model
    model = GeoEmbeddingModel().to(device)
    
    # Dataset and DataLoader
    dataset = GeoDataset(image_paths=image_paths, coordinates=coordinates, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    # Generate embeddings
    print("Generating custom embeddings...")
    embeddings = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, _ in track(dataloader, description="Processing images..."):
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy().astype(np.float32))  # Move to CPU and convert to float32
    embeddings = np.vstack(embeddings)

    # Save the embeddings and image paths
    np.save('Roy/ML/embeddings.npy', embeddings)
    np.save('Roy/ML/image_paths.npy', np.array(image_paths))
    
    # save the model
    torch.save(model.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")

# Extract the latitude and longitude from each image path
coordinates = np.array([extract_coordinates(path) for path in image_paths])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, coordinates, test_size=4000/embeddings.shape[0], shuffle=True, random_state=42)

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout0 = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.gelu1 = nn.GELU()
        
        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.gelu2 = nn.GELU()
        
        self.fc3 = nn.Linear(512, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.gelu3 = nn.GELU()
        
        self.fc4 = nn.Linear(256, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.gelu4 = nn.GELU()
        
        self.fc5 = nn.Linear(128, 32)
        self.batch_norm5 = nn.BatchNorm1d(32)
        self.gelu5 = nn.GELU()
        
        self.fc6 = nn.Linear(32, 16)
        self.batch_norm6 = nn.BatchNorm1d(16)
        self.gelu6 = nn.GELU()

        self.fc7 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout0(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.gelu3(x)

        x = self.fc4(x)
        x = self.batch_norm4(x)
        x = self.gelu4(x)

        x = self.fc5(x)  
        x = self.batch_norm5(x)
        x = self.gelu5(x)

        x = self.fc6(x)
        x = self.batch_norm6(x)
        x = self.gelu6(x)

        x = self.fc7(x)    

        return x

# Initialize the model and optimizer
geo_predictor = GeoPredictorNN().to(device)
optimizer = optim.AdamW(geo_predictor.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-6)
scaler = GradScaler()  # For mixed precision training

# Prepare DataLoader for training
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

# Haversine loss function
def degrees_to_radians(deg):
    return deg * 0.017453292519943295  # Pi/180

def haversine_loss(coords1, coords2):
    lat1, lon1 = coords1[:, 0], coords1[:, 1]
    lat2, lon2 = coords2[:, 0], coords2[:, 1]

    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))

    distance = 6371.01 * c
    return distance.mean()

# Training loop
epochs = 250
losses = []
val_losses = []
for epoch in track(range(epochs), description="Training the model..."):
    geo_predictor.train()
    running_loss = 0.0
    for batch in train_loader:
        embeddings_batch, coords_batch = batch
        embeddings_batch, coords_batch = embeddings_batch.float().to(device), coords_batch.float().to(device)

        optimizer.zero_grad()

        # Mixed Precision (autocast)
        with autocast():
            outputs = geo_predictor(embeddings_batch)
            loss = haversine_loss(outputs, coords_batch)

        # Backprop and optimizer step with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    losses.append(running_loss / len(train_loader))

    # Validation step
    geo_predictor.eval()
    with torch.no_grad():
        with autocast():
            val_loss = haversine_loss(geo_predictor(torch.tensor(X_test).float().to(device)), torch.tensor(y_test).float().to(device))
        val_losses.append(val_loss.item())

    # Scheduler step
    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss.item():.4f}")

# Save the trained model
torch.save(geo_predictor.state_dict(), 'Roy/ML/Saved_Models/geo_predictor_nn.pth')

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, color='skyblue')
plt.plot(val_losses, color='orange')
plt.title('Training Loss vs Validation Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
