# -------------------------------
# FULL SCRIPT: Training + Evaluation
# -------------------------------
import os, threading, warnings, time, math
import tkinter as tk
from tkinter import font
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from rich.progress import track
from PIL import Image
from torchvision import models, transforms

warnings.filterwarnings("ignore")

def create_loss_window():
    root = tk.Tk()
    root.title("Validation Loss Monitor")
    large_font = font.Font(family="Helvetica", size=72, weight="bold")
    loss_label = tk.Label(root, text="Model: N/A\nEpoch: N/A\nVal Loss: N/A\nLowest Loss: N/A", font=large_font, bg="white")
    loss_label.pack(padx=20, pady=20)
    return root, loss_label

def update_loss_label(loss_label, epoch, new_loss, min_val_loss, modeln):
    points = np.floor(5000 * np.exp(-1 * new_loss / 2000))
    loss_label.config(text=f"Model: {modeln+1}\nEpoch: {epoch}\nVal Loss: {new_loss:.1f} km\nLowest Loss: {min_val_loss:.1f} km\nGeoguessr Points: {int(points)}")
    loss_label.update_idletasks()

def launch_loss_window(event):
    global loss_root, loss_label
    loss_root, loss_label = create_loss_window()
    event.set()
    loss_root.mainloop()

# GUI thread
window_event = threading.Event()
threading.Thread(target=launch_loss_window, args=(window_event,), daemon=True).start()
window_event.wait()

# -------------------
# MODEL DEFINITIONS
# -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [2048, 1024, 512, 256, 128, 32, 16, 2]
        layers = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i+1]), nn.BatchNorm1d(sizes[i+1]), nn.GELU(), nn.Dropout(0.2)]
        layers += [nn.Linear(sizes[-2], sizes[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
    def forward(self, x):
        return self.backbone(x).view(x.size(0), -1)

# -------------------
# TRAINING
# -------------------
def extract_coordinates(path):
    lat = float(path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(path.split('_')[1])
    return lat, lon

def degrees_to_radians(deg): return deg * np.pi / 180

def haversine_loss(pred, true):
    lat1, lon1, lat2, lon2 = map(degrees_to_radians, (true[:, 0], true[:, 1], pred[:, 0], pred[:, 1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    return (6371.01 * 2 * torch.arcsin(torch.sqrt(a))).mean()

embed_file = 'Roy/ML/embeddings.npy'
img_file = 'Roy/ML/image_paths.npy'
embeddings = np.load(embed_file).astype(np.float32)
image_paths = np.load(img_file)
coords = np.array([extract_coordinates(p) for p in image_paths])

X_train, X_test, y_train, y_test = train_test_split(embeddings, coords, test_size=4000/len(embeddings), random_state=0)
X_train, X_test = map(lambda x: torch.tensor(x).float().to(device), (X_train, X_test))
y_train, y_test = map(lambda x: torch.tensor(x).float().to(device), (y_train, y_test))

for modelnumber in range(1):
    print(f"Model number: {modelnumber}")
    geo_predictor = GeoPredictorNN().to(device)

    criterion = haversine_loss
    optimizer = optim.AdamW(geo_predictor.parameters(), lr=1e-4, weight_decay=1e-4, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.95, threshold=0.01, verbose=True)
    batch_size = 1024
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    max_epochs = 50
    min_val_loss, val_losses, losses = 1e8, [], []
    for epoch in track(range(max_epochs), description="Training"):
        geo_predictor.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(geo_predictor(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))

        geo_predictor.eval()
        with torch.no_grad():
            val_loss = criterion(geo_predictor(X_test), y_test).item()
        val_losses.append(val_loss)
        update_loss_label(loss_label, epoch+1, val_loss, min_val_loss, modelnumber)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
        scheduler.step(val_loss)

    final_name = f'geo_predictor_nn_{max_epochs}e_{batch_size}b_{int(min_val_loss)}k'
    final_path = f'Roy/ML/Fully_Auto/Storage/Models/{final_name}.pth'
    torch.save(geo_predictor.state_dict(), final_path)
loss_root.quit()

# -------------------
# EVALUATION
# -------------------
def load_and_transform_images(paths):
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs).to(device)

def haversine(coord1, coord2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def geoguessr_points_formula(err):
    return 5000 if err < 0.15 else np.floor(5000 * math.exp(-err / 2000))

image_dir = 'Roy/Test_Images'
image_paths = sorted([f"{image_dir}/{f}" for f in os.listdir(image_dir)
                      if f.endswith('.jpg') and "Game" in f])

real_coords = np.array([
    [59.2641988, 10.4276279],
    [1.4855156, 103.8675691],
    [54.9926562, -1.6732242],
    [19.4744679, -99.1973953],
    [58.6133469, 49.6274857]
])

image_tensor_batch = load_and_transform_images(image_paths)

embedding_model = GeoEmbeddingModel().to(device)
embedding_model.load_state_dict(torch.load(
    'Roy/ML/Fully_Auto/Storage/geo_embedding_model_r152_normal.pth', map_location=device))
embedding_model.eval()

with torch.no_grad():
    embeddings = embedding_model(image_tensor_batch).cpu()

model_dir = 'Roy/ML/Fully_Auto/Storage/Models'
model_names = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
leaderboard = []
start_time = time.time()

with torch.no_grad():
    for name in model_names:
        predictor = GeoPredictorNN().to(device)
        predictor.load_state_dict(torch.load(f"{model_dir}/{name}", map_location=device))
        predictor.eval()

        preds = predictor(embeddings.to(device)).cpu().numpy()
        preds[:, 0] = (preds[:, 0] + 90) % 180 - 90
        preds[:, 1] = (preds[:, 1] + 180) % 360 - 180

        errors = [haversine(gt, pred) for gt, pred in zip(real_coords, preds)]
        total_points = sum(geoguessr_points_formula(e) for e in errors)
        leaderboard.append([name, total_points])

leaderboard.sort(key=lambda x: -x[1])
print("leaderboard = [")
for model in leaderboard:
    print(f"    ['{model[0]}', {int(model[1])}],")
print("]")

with open("Roy/ML/Fully_Auto/Storage/leaderboard.txt", "w") as f:
    f.write("[\n")
    for model in leaderboard:
        f.write(f"    ['{model[0]}', {int(model[1])}],\n")
    f.write("]\n")

print(f"Execution time: {time.time() - start_time:.1f} seconds")
