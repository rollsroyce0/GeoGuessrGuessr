import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import geopandas as gpd
import matplotlib.pyplot as plt
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
    plt.legend(); plt.show(block=False); plt.pause(0.1); plt.close()


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
    low_points_backup = []
    errors = []
    low_errors = []
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
        else:
            low_points_backup.append(pts)
            low_errors.append(errs)
        
        results.append((fname, total_pts, preds.tolist()))
        full_results.append((fname, total_pts, preds.tolist()))
        # Sort results by total points in descending order and keep the top 3 models
        results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        #print(f"{fname}: {total_pts} pts")

    if len(points_backup) == 0:
        print("No models scored above 10,000 points. Please check your models.")
        points_backup = low_points_backup
        errors = low_errors
    backups = []
    for fname, _, preds in full_results:
        #print(f"{fname}: {total_pts}")
        backups.append([fname, preds])


    
    # Load model weights from Best_overall_models.txt
    weights_file = "Roy/Test_Images/Best_overall_models.txt"
    model_weights = {}
    if os.path.exists(weights_file):
        with open(weights_file, "r") as f:
            for line in f:
                if line.startswith('leaderboard') or line.startswith('[') or line.startswith(']'):
                    continue
                parts = line.replace("'", "").replace("[", "").replace("]", "").replace(" ", "")
                part = parts.strip().split(",")
                model_name = part[0].strip()
                score = int(part[1].strip())
                if score >=25:
                    model_weights[model_name] = score
                
    #print("Model weights loaded from file:", model_weights)
    
    # Sort the models by their weights
    sorted_models = sorted(model_weights.items(), key=lambda x: x[1], reverse=True)
    print("number of Sorted models by weights:", len(sorted_models))
    # reduce backups to the models found in sorted_models
    backups = [b for b in backups if b[0] in dict(sorted_models)]
    
    print("number of models in backups:", len(backups))
    
    # Sort backups by the model weights
    backups = sorted(backups, key=lambda x: model_weights[x[0]], reverse=True)
    print("number of models in backups after sorting:", len(backups))
    
    # Since we have a batch of 5 predictions with two coordinates each, we need to average each one of them between all models using the weights
    # Initialize the final predictions
    avg_preds = np.zeros((5, 2))
    # Initialize the total weights
    total_weights = 0
    # Loop over the backups and their weights
    for model, preds in backups:
        # Get the weight of the model
        weight = model_weights[model]
        print(f"Model: {model}, Weight: {weight}")
        # Add the weighted predictions to the final predictions
        avg_preds += np.array(preds) * weight**2
        # Add the weight to the total weights
        total_weights += weight**2
    print("Total weights:", total_weights)
    # Normalize the final predictions by the total weights
    avg_preds /= total_weights
    print("Average predictions:", avg_preds)


    final_errs = haversine_batch(real_coords, avg_preds)
    final_pts = [geoguessr_points(e) for e in final_errs]
    print("Final points for each image:", final_pts)
    print("Final total:", sum(final_pts))
    print("Highest points for each image:", highest_points)
    print("Highest total:", sum(highest_points))
    
    # reduce backups to only the predictions
    backups_scores = [b[1] for b in backups]
    
    print("Backups scores shape:", np.array(backups_scores).shape)
    backups_scores = np.array(backups_scores)
    #print(backups_scores[:, 0])
    
    
    
    for i, path in enumerate(img_paths):
        plot_coordinates_on_map(avg_preds[i], real_coords[i], backups_scores[:, i], path)
    print(f"Time elapsed: {time.time()-start:.2f}s")
    return results, full_results, total_points_backup, errors


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