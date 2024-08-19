import os
import torch
import numpy as np
import joblib
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

def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

def normalize_coordinates(coords):
    lat, lon = coords[:, 0], coords[:, 1]
    lat = (lat + 90) / 180
    lon = (lon + 180) / 360
    return np.stack([lat, lon], axis=1)

def denormalize_coordinates(coords):
    lat, lon = coords[:, 0], coords[:, 1]
    lat = lat * 180 - 90
    lon = lon * 360 - 180
    return np.stack([lat, lon], axis=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

location = "D:/GeoGuessrGuessr/geoguesst"
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super(GeoEmbeddingModel, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x

if os.path.exists('Roy/ML/embeddings.npy'):
    embeddings = np.load('Roy/ML/embeddings.npy').astype(np.float32)
    image_paths = np.load('Roy/ML/image_paths.npy')
else:
    image_paths = [os.path.join(location, img_file) for img_file in os.listdir(location)]
    coordinates = np.array([extract_coordinates(path) for path in image_paths])

    model = GeoEmbeddingModel().to(device)
    
    dataset = GeoDataset(image_paths=image_paths, coordinates=coordinates, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model.pth'):
        model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model.pth'))

    print("Generating custom embeddings...")
    embeddings = []
    model.eval()
    with torch.no_grad():
        for images, _ in track(dataloader, description="Processing images..."):
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy().astype(np.float32))
    embeddings = np.vstack(embeddings)

    np.save('Roy/ML/embeddings.npy', embeddings)
    np.save('Roy/ML/image_paths.npy', np.array(image_paths))
    torch.save(model.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")

coordinates = np.array([extract_coordinates(path) for path in image_paths])

def assign_continent(lat, lon):
    if lat < 30 and -30 <= lon <= 60:
        return 0
    elif lat > -13 and lon > 45:
        return 1
    elif -50 < lat < -13 and 110 <= lon <= 180:
        return 2
    elif lat > 12 and -130 <= lon <= -30:
        return 3
    elif lat < 12 and -90 <= lon <= -30:
        return 4
    elif lat > 30 and -30 <= lon <= 45:
        return 5
    else:
        return 6

continent_labels = np.array([assign_continent(lat, lon) for lat, lon in coordinates])

X_train, X_test, y_train, y_test, train_labels, test_labels = train_test_split(
    embeddings, coordinates, continent_labels, test_size=5000/embeddings.shape[0]
)

y_train_normalized = normalize_coordinates(y_train)
y_test_normalized = normalize_coordinates(y_test)

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super(GeoPredictorNN, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.gelu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x

geo_predictor = GeoPredictorNN().to(device)

def degrees_to_radians(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180

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

criterion = haversine_loss
optimizer = optim.AdamW(geo_predictor.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

train_loader = DataLoader(list(zip(X_train, y_train_normalized)), batch_size=16, shuffle=True)

epochs = 20
losses = []
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

    scheduler.step()
    losses.append(running_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

torch.save({
    'epoch': epoch,
    'model_state_dict': geo_predictor.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'Roy/ML/Saved_Models/geo_predictor_nn.pth')

plt.figure(figsize=(10, 6))
plt.plot(losses, color='skyblue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

def haversine_distance(coords1, coords2):
    return geopy.distance.geodesic(coords1, coords2).kilometers

def predict_coordinates_nn(embedding, geo_predictor):
    geo_predictor.eval()
    with torch.no_grad():
        embedding = torch.tensor(embedding).float().unsqueeze(0).to(device)
        predicted_coords = geo_predictor(embedding).cpu().numpy()[0]
    return predicted_coords

def evaluate_nn_model(X_test, y_test, geo_predictor):
    y_pred = np.array([predict_coordinates_nn(embedding, geo_predictor) for embedding in track(X_test, description="Predicting coordinates...")])

    y_test = denormalize_coordinates(y_test)
    y_pred = denormalize_coordinates(y_pred)

    distances = np.array([haversine_distance(y_test[i], y_pred[i]) for i in range(len(y_test))])

    print(f"Mean distance: {np.mean(distances):.2f} km")
    print(f"Median distance: {np.median(distances):.2f} km")
    print(f"Max distance: {np.max(distances):.2f} km")
    print(f"Min distance: {np.min(distances):.2f} km")

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Haversine Distances (km)')
    plt.xlabel('Distance (km)')
    plt.ylabel('Frequency')
    plt.show()

evaluate_nn_model(X_test, y_test_normalized, geo_predictor)
