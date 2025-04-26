import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import time
import warnings
warnings.filterwarnings("ignore")

# Custom Model to generate embeddings
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

# Custom model for predicting coordinates with original layer names
class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [2048, 1024, 512, 256, 128, 32, 16, 2]
        layers = []
        for i in range(len(sizes) - 2):
            layers += [
                nn.Linear(sizes[i], sizes[i+1]),
                nn.BatchNorm1d(sizes[i+1]),
                nn.GELU(),
                nn.Dropout(0.2)
            ]
        layers += [nn.Linear(sizes[-2], sizes[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Image loading and transform
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_images(folder, testtype):
    imgs, paths = [], []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith('.jpg') and testtype in fn:
            img = Image.open(os.path.join(folder, fn)).convert('RGB')
            imgs.append(transform(img))
            paths.append(os.path.join(folder, fn))
    return torch.stack(imgs), paths

# Vectorized haversine and scoring

def haversine_batch(coords1, coords2):
    R = 6371.0
    lat1 = np.radians(coords1[:,0]); lon1 = np.radians(coords1[:,1])
    lat2 = np.radians(coords2[:,0]); lon2 = np.radians(coords2[:,1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def geoguessr_points(error):
    return 5000 if error < 0.15 else np.floor(5000 * np.exp(-error/2000))

# Plot function unchanged
def plot_coordinates_on_map(pred, real, backups, path):
    plt.figure(figsize=(10,8))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=plt.gca(), linewidth=1)
    plt.scatter(pred[1], pred[0], c='red', label='Predicted', s=100)
    plt.scatter(real[1], real[0], c='green', label='Real', s=100, alpha=0.5)
    for b in backups:
        plt.scatter(b[1], b[0], c='blue', s=80, alpha=0.5)
    allc = np.vstack([*backups, pred, real])
    lat_min, lat_max = allc[:,0].min(), allc[:,0].max()
    lon_min, lon_max = allc[:,1].min(), allc[:,1].max()
    mlat = (lat_max - lat_min)*0.5+2; mlon = (lon_max - lon_min)*0.5+2
    plt.xlim(lon_min-mlon, lon_max+mlon); plt.ylim(lat_min-mlat, lat_max+mlat)
    plt.title(f"Map: {os.path.basename(path)}"); plt.xlabel('Lon'); plt.ylabel('Lat')
    plt.legend(); plt.show(block=False); plt.pause(1); plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess images once
    testtype = 'Game' #'Validation' or 'Game' or 'Verification' or 'Super'
    images, img_paths = load_images('Roy/Test_Images', testtype)
    images = images.to(device)

    # Real coordinates
    real_coords_Game = np.array([[59.2642, 10.4276], [1.4855, 103.8676], [54.9927, -1.6732], [19.4745, -99.1974], [58.6133, 49.6275]])
    real_coords_Valid = np.array([[43.3219114, -5.5783907], [23.0376137, 72.5819308], [55.9300025, -3.2678762], [51.9187417, 4.4957128], [40.6000729, -74.3125485]])
    real_coords_Verification = np.array([[48.1787242,16.4149478], [39.3544037,-76.4284282], [12.6545729,77.4269159], [53.5361597,-113.470894], [65.9408919,12.2171864]])
    real_coords_Super = np.array([[47.0676173,12.5318788], [45.8186432,-63.4844332], [41.8610051,12.5368213], [-6.3320979,106.8518361], [35.6061998,-77.3731937]])
    
    if testtype == 'Game':
        real_coords = real_coords_Game
    elif testtype == 'Validation':
        real_coords = real_coords_Valid
    elif testtype == 'Verification':
        real_coords = real_coords_Verification
    elif testtype == 'Super':
        real_coords = real_coords_Super
    else:
        raise ValueError("Invalid test type. Choose 'Game', 'Validation', 'Super', or 'Verification'.")
    
    # Initialize embedding model
    embed_model = GeoEmbeddingModel().to(device).eval()
    embed_model.load_state_dict(torch.load('Roy/ML/Shotgun/Storage/geo_embedding_model_r152_normal.pth', map_location=device))

    # Precompute embeddings
    with torch.no_grad():
        embeddings = embed_model(images).cpu()

    results = []
    full_results = []
    points_backup = []
    errors = []
    start = time.time()
    highest_points = [0,0,0,0,0]
    total_points_backup = []

    # Loop over predictor weights
    for fname in sorted(os.listdir('Roy/ML/Shotgun/Storage/Models')):
        if 'embedding' in fname or 'lowest' in fname or not fname.endswith('.pth'):
            continue

        predictor = GeoPredictorNN().to(device).eval()
        predictor.load_state_dict(torch.load(f'Roy/ML/Shotgun/Storage/Models/{fname}', map_location=device))

        with torch.no_grad():
            preds = predictor(embeddings.to(device)).cpu().numpy()
            preds[:,0] = (preds[:,0]+90)%180 - 90
            preds[:,1] = (preds[:,1]+180)%360 - 180

        errs = haversine_batch(real_coords, preds)
        pts = [geoguessr_points(e) for e in errs]
        total_pts = sum(pts)
        total_points_backup.append(total_pts)
        # for each picture, check if the points are higher than the previous highest points
        for i, p in enumerate(pts):
            if p > highest_points[i]:
                highest_points[i] = p
                
        if total_pts > 10000:
            points_backup.append(pts)
            errors.append(errs)
        results.append((fname, total_pts, preds.tolist()))
        full_results.append((fname, total_pts, preds.tolist()))
        # Sort results by total points in descending order and keep the top 3 models
        results = sorted(results, key=lambda x: x[1], reverse=True)[:3]
        print(f"{fname}: {total_pts} pts")

    print("Top 3 models:")
    for i, (fname, total_pts, preds) in enumerate(results):
        print(f"{i+1}: {fname} - {total_pts} pts")
        #print(preds)
    # Save the testtype and the best three models to a file
    # Check if the file exists, if not create it
    if not os.path.exists(f'Roy/ML/Shotgun/Best_models_{testtype}.txt'):
        with open(f'Roy/ML/Shotgun/Best_models_{testtype}.txt', 'w') as f:
            f.write("Best models for each test type:\n")
    with open(f'Roy/ML/Shotgun/Best_models_{testtype}.txt', 'a') as f:
        
        f.write(f"{testtype}: {results[0][0]}, {results[1][0]}, {results[2][0]}\n")
    
    backups = list(zip(*[r[2] for r in results]))
    avg_preds = np.mean(np.array(backups), axis=1)

    final_errs = haversine_batch(real_coords, avg_preds)
    final_pts = [geoguessr_points(e) for e in final_errs]
    for i, err in enumerate(final_errs):
        print(f"Final error for {i}: {err} km")
    print("Final points for each image:", final_pts)
    print("Final total:", sum(final_pts))
    print("Highest points for each image:", highest_points)
    print("Highest total:", sum(highest_points))
    
    # Calculate a Difficulty score for each image based on the standard deviation of the predictions
    errors = np.sort(errors, axis=0)[:-10] # Remove the 10 highest errors or each individual image disregarding model order
    points_backup = np.sort(points_backup, axis=0)[:10]
    difficulty_scores = np.std(errors, axis=0) + 0.4*np.mean(errors, axis=0) # Add the mean to the std to get a more accurate score
    #print("Errors:", errors)
    print("Difficulty scores:", difficulty_scores)
    # Normalize these on a scale 0-10, where an std dev of 2500 would be a difficulty of 10 and 0 would be 0. However this is not a linear scale, so we will use a logarithmic scale.
    # We will use a base of 10, so that 10^0 = 1 and 10^1 = 10. This means that a difficulty of 0 would be 0 and a difficulty of 10 would be 10.
    # We will also use a minimum difficulty of 1, so that we don't get negative scores.
    # A Difficulty of 10 means the std is 4500km or above
    
    
    difficulty_scores = np.log10(difficulty_scores/1000 + 1) * 13
    difficulty_scores = np.clip(difficulty_scores, 0, 10)
    difficulty_scores = np.round(difficulty_scores, 3)
    print("Difficulty scores normalized:", difficulty_scores)
    print("Difficulty scores for each image:", difficulty_scores)
    print("Average difficulty score of this round:", np.round(np.mean(difficulty_scores), 3))
    # add the average difficutly score for the test type to a file
    with open(f'Roy/ML/Shotgun/Difficulty_scores.txt', 'a') as f:
        f.write(f"{testtype}: {np.round(np.mean(difficulty_scores), 3)}, Highest: {np.round(np.max(difficulty_scores), 3)}, Lowest: {np.round(np.min(difficulty_scores), 3)}\n")
        
        # remove any duplicate lines (it is a duplicate, if the first 5 characters are the same)
    with open(f'Roy/ML/Shotgun/Difficulty_scores.txt', 'r') as f:
        lines = f.readlines()
    
    # remove duplicates by checking the first 5 characters of each line
    seen = set()
    lines = [line for line in lines if not (line[:5] in seen or seen.add(line[:5]))]
    
    # write the lines back to the file
    
    with open(f'Roy/ML/Shotgun/Difficulty_scores.txt', 'w') as f:
        f.writelines(lines)

    for i, path in enumerate(img_paths):
        plot_coordinates_on_map(avg_preds[i], real_coords[i], backups[i], path)

    print(f"Time elapsed: {time.time()-start:.2f}s")
