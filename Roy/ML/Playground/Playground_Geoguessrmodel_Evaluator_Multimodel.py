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
        dims = [2048, 1024, 512, 256, 128, 32, 16]
        for i in range(len(dims)-1):
            in_dim, out_dim = dims[i], dims[i+1]
            setattr(self, f'fc{i+1}', nn.Linear(in_dim, out_dim))
            setattr(self, f'batch_norm{i+1}', nn.BatchNorm1d(out_dim))
            setattr(self, f'gelu{i+1}', nn.GELU())
            # smaller dropout on last block
            dropout_rate = 0.1 if i == len(dims)-2 else 0.2
            setattr(self, f'dropout{i+1}', nn.Dropout(dropout_rate))
        # final layer
        self.fc7 = nn.Linear(16, 2)

    def forward(self, x):
        # sequentially apply each block
        for i in range(1, 7):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'dropout{i}')(x)
            x = getattr(self, f'batch_norm{i}')(x)
            x = getattr(self, f'gelu{i}')(x)
        x = self.fc7(x)
        return x

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
    plt.legend(); plt.show(block=False); plt.pause(10)
    plt.close()

def main(testtype=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess images once
    if testtype is None:
        testtype = input("Enter test type (Game, Validation, Super, Verification, Ultra, Extreme, Chrome): ")
    if testtype not in ['Game', 'Validation', 'Super', 'Verification', 'Ultra', 'Extreme', 'Chrome', 'World', 'Task', 'Enlarged', 'Exam', 'Google']:
        raise ValueError("Invalid test type. Choose 'Game', 'Validation', 'Super', 'Ultra', or any other.")
    images, img_paths = load_images('Roy/Test_Images', testtype)
    images = images.to(device)

    # Real coordinates
    real_coords_Game = np.array([[59.2642, 10.4276], [1.4855, 103.8676], [54.9927, -1.6732], [19.4745, -99.1974], [58.6133, 49.6275]])
    real_coords_Valid = np.array([[43.3219114, -5.5783907], [23.0376137, 72.5819308], [55.9300025, -3.2678762], [51.9187417, 4.4957128], [40.6000729, -74.3125485]])
    real_coords_Verification = np.array([[48.1787242,16.4149478], [39.3544037,-76.4284282], [12.6545729,77.4269159], [53.5361597,-113.470894], [65.9408919,12.2171864]])
    real_coords_Super = np.array([[47.0676173,12.5318788], [45.8186432,-63.4844332], [41.8610051,12.5368213], [-6.3320979,106.8518361], [35.6061998,-77.3731937]])
    real_coords_Ultra = np.array([[45.5097937,27.8273201], [41.71066,-93.7363551], [-31.387591,-57.9646316], [-37.8980561,144.626041], [54.3667423,-6.7718667]])
    real_coords_Extreme = np.array([[8.6521503,81.0067919], [46.2477428,-80.4337085], [60.3055298,56.971439], [40.8337969,-74.0825815], [43.1914376,17.3801763]])
    real_coords_Chrome = np.array([[34.5230872,-86.9699949], [52.2929603,4.6668573], [52.585936,-0.2501907], [32.5221938,-82.9127378], [39.7692443,30.5314142]])
    real_coords_World = np.array([[-6.8146562,-38.6533882], [12.1391977,-68.9490383], [59.4227739,15.8038038], [51.5529906,-0.4758671], [14.3329551,99.6477487]])
    real_coords_Task = np.array([[34.2468633,-82.2092303], [49.935202,5.4581067], [43.9435807,12.4477353], [48.08332,-0.6451421], [53.3559593,55.9645235]])
    real_coords_Enlarged = np.array([[-34.8295223,-58.8707693], [40.4369798,-3.6859228], [-54.1257734,-68.0709486], [48.9828428,12.6387341], [45.9312686,-82.4707373]])
    real_coords_Exam = np.array([[-4.1237242,-38.3705862], [40.1161881,-75.1248975], [35.1362241,136.7419345], [41.6557297,-91.5466424], [-47.0777189,-72.1646972]])
    real_coords_Google = np.array([[59.407269,15.415694], [52.5644145,-110.8206357], [-36.8700509,174.6481411], [37.9270951,-122.53026], [28.6397445,77.2929918]])   
    if testtype == 'Game':
        real_coords = real_coords_Game
    elif testtype == 'Validation':
        real_coords = real_coords_Valid
    elif testtype == 'Verification':
        real_coords = real_coords_Verification
    elif testtype == 'Super':
        real_coords = real_coords_Super
    elif testtype == 'Ultra':
        real_coords = real_coords_Ultra
    elif testtype == 'Extreme':
        real_coords = real_coords_Extreme
    elif testtype == 'Chrome':
        real_coords = real_coords_Chrome
    elif testtype == 'World':
        real_coords = real_coords_World
    elif testtype == 'Task':
        real_coords = real_coords_Task
    elif testtype == 'Enlarged':
        real_coords = real_coords_Enlarged
    elif testtype == 'Exam':
        real_coords = real_coords_Exam
    elif testtype == 'Google':
        real_coords = real_coords_Google
    else:
        raise ValueError("Invalid test type. Choose 'Game', 'Validation', 'Super', or 'Verification', or 'Ultra', or 'Extreme', or 'Chrome', or 'World', or 'Task', or 'Enlarged', or 'Exam'.")
    
    # Initialize embedding model
    embed_model = GeoEmbeddingModel().to(device).eval()
    embed_model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth', map_location=device))

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
    for fname in sorted(os.listdir('Roy/ML/Saved_Models')):
        if 'embedding' in fname or 'lowest' in fname or not fname.endswith('.pth'):
            continue

        predictor = GeoPredictorNN().to(device).eval()
        predictor.load_state_dict(torch.load(f'Roy/ML/Saved_Models/{fname}', map_location=device))

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
        results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        #print(f"{fname}: {total_pts} pts")

    print(f"Top 10 models for {testtype}:")
    for i, (fname, total_pts, preds) in enumerate(results):
        print(f"{i+1}: {fname} - {total_pts} pts")
        #print(preds)
    # Save the testtype and the best three models to a file
    # Check if the file exists, if not create it
    if not os.path.exists(f'Roy/Test_Images/Best_models_{testtype}.txt'):
        # Throw an error if the file does not exist
        raise FileNotFoundError(f"File Roy/Test_Images/Best_models_{testtype}.txt does not exist")
    # remove all text from the file
    with open(f'Roy/Test_Images/Best_models_{testtype}.txt', 'r+') as f:
        #one=1
        # remove everything from the file
        f.truncate(0)

    
    with open(f'Roy/Test_Images/Best_models_{testtype}.txt', 'a') as f:
        #one=1
        # remove everything from the file
        
        f.write("Best 10 models for each test type:\n")
        f.write(f"{testtype}: {results[0][0]}, {results[1][0]}, {results[2][0]}, {results[3][0]}, {results[4][0]}, {results[5][0]}, {results[6][0]}, {results[7][0]}, {results[8][0]}, {results[9][0]}\n")
    
    backups = list(zip(*[r[2] for r in results]))
    avg_preds = np.mean(np.array(backups), axis=1)

    final_errs = haversine_batch(real_coords, avg_preds)
    final_pts = [geoguessr_points(e) for e in final_errs]
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
    difficulty_scores = difficulty_scores**2
    difficulty_scores = np.round(difficulty_scores, 3)
    print("Difficulty scores normalized:", difficulty_scores)
    print("Difficulty scores for each image:", difficulty_scores)
    print("Average difficulty score of this round:", np.round(np.mean(difficulty_scores), 3))
    # add the average difficutly score for the test type to a file
    if np.mean(difficulty_scores) == np.nan:
        difficulty_scores = 100
    with open(f'Roy/Test_Images/Difficulty_scores.txt', 'a') as f:
        f.write(f"{testtype}: {np.round(np.mean(difficulty_scores), 3)}, Highest: {np.round(np.max(difficulty_scores), 3)}, Lowest: {np.round(np.min(difficulty_scores), 3)}\n")
        
        # remove any duplicate lines (it is a duplicate, if the first 5 characters are the same)
    with open(f'Roy/Test_Images/Difficulty_scores.txt', 'r') as f:
        lines = f.readlines()
    
    # remove duplicates by checking the first 5 characters of each line
    seen = set()
    lines = [line for line in lines if not (line[:5] in seen or seen.add(line[:5]))]
    
    # write the lines back to the file
    
    with open(f'Roy/Test_Images/Difficulty_scores.txt', 'w') as f:
        f.writelines(lines)

    for i, path in enumerate(img_paths):
        plot_coordinates_on_map(avg_preds[i], real_coords[i], backups[i], path)

    print(f"Time elapsed: {time.time()-start:.2f}s")
    return results, full_results, total_points_backup, errors, difficulty_scores


if __name__ == "__main__":
    start_time = time.time()
    testtype = 'All' #'Validation' or 'Game' or 'Verification' or 'Super' or 'All'
    if testtype == 'All':
        for testtype in ['Game', 'Validation', 'Super', 'Verification', 'Ultra', 'Extreme','Chrome', 'World', 'Task', 'Enlarged', 'Exam', 'Google']:
            print("\n----------------------------------------------------------------------\n")
            main(testtype)
    else:
        main(testtype)
        #main() # Uncomment this line to run the main function without any arguments and accept user input
    print(f"Execution time: {time.time() - start_time} seconds")