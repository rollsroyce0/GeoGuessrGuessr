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
from rich.progress import track
warnings.filterwarnings("ignore")

from Second_Level_ML.generate_coordinates import get_real_coordinates, list_test_types
global list_of_maps
list_of_maps = list_test_types()

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


def main(testtype=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testtype: {testtype}")
    # Load and preprocess images once
    if testtype is None:
        testtype = input("Enter test type (Game, Validation, Super, Verification, Ultra, Extreme, Chrome): ")
    if testtype not in list_of_maps:
        raise ValueError("Invalid test type. Choose 'Game', 'Validation', 'Super', 'Ultra', or any other.")
    images, img_paths = load_images('Roy/Test_Images', testtype)
    images = images.to(device)

    # Real coordinates
    
    real_coords = get_real_coordinates(testtype)
    
    # Initialize embedding model
    embed_model = GeoEmbeddingModel().to(device).eval()
    embed_model.load_state_dict(torch.load('Roy/ML/Saved_Models/Best_geo_embedding_model_r152_normal.pth', map_location=device))

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
    names_bad = []
    print(f"Number of predictor models: {len(os.listdir('Roy/ML/Saved_Models'))}")
    print(f"Number of checkpoint models: {len(os.listdir('Roy/ML/Saved_Models_Checkpoint/Checkpoint_Models_NN'))}")

    # Loop over predictor weights
    for fname in track(sorted(os.listdir('Roy/ML/Saved_Models'))):
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

    average_points = np.mean(np.array(points_backup), axis=0) if points_backup else np.zeros(len(real_coords))
    # Loop over all models in a second folder named 'Roy/ML/Saved_Models_Checkpoint/Checkpoint_Models_NN/'
    for fname in track(sorted(os.listdir('Roy/ML/Saved_Models_Checkpoint/Checkpoint_Models_NN'))):
        if 'embedding' in fname or 'lowest' in fname or not fname.endswith('.pth'):
            continue

        predictor = GeoPredictorNN().to(device).eval()
        predictor.load_state_dict(torch.load(f'Roy/ML/Saved_Models_Checkpoint/Checkpoint_Models_NN/{fname}', map_location=device))

        with torch.no_grad():
            preds = predictor(embeddings.to(device)).cpu().numpy()
            preds[:,0] = (preds[:,0]+90)%180 - 90
            preds[:,1] = (preds[:,1]+180)%360 - 180

        errs = haversine_batch(real_coords, preds)
        pts = [geoguessr_points(e) for e in errs]
        total_pts = sum(pts)
        if total_pts < average_points.sum():
            # delete the model if it is not better than the average points
            #print(f"{fname} scored {total_pts} pts, which is lower than the average points {average_points.sum()}. Deleting model.")
            names_bad.append(fname)
            #os.remove(f'Roy/ML/Saved_Models_Checkpoint/Checkpoint_Models_NN/{fname}')
            continue
        
        total_points_backup.append(total_pts)
        # for each picture, check if the points are higher than the previous highest points
        for i, p in enumerate(pts):
            if p > highest_points[i]:
                highest_points[i] = p
                
        if total_pts > 10000:
            #print(f"{fname}: {total_pts} pts")
            points_backup.append(pts)
            errors.append(errs)
        else:
            low_points_backup.append(pts)
            low_errors.append(errs)

        results.append((fname, total_pts, preds.tolist()))
        full_results.append((fname, total_pts, preds.tolist()))
        # Sort results by total points in descending order and keep the top 3 models
        results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    
    
    if len(points_backup) == 0:
        print("No models scored above 10,000 points. Please check your models.")
        points_backup = low_points_backup
        errors = low_errors
    print(f"Top 10 models for {testtype}:")
    for i, (fname, total_pts, preds) in enumerate(results):
        print(f"{i+1}: {fname} - {total_pts} pts")
        #print(preds)
    
    # if any of the checkpoint models appear in the results, move the model outside the checkpoint folder
    for fname, total_pts, preds in results:
        if 'checkpoint' in fname:
            new_fname = fname.replace('checkpoint', 'check')
            os.rename(f'Roy/ML/Saved_Models_Checkpoint/Checkpoint_Models_NN/{fname}', f'Roy/ML/Saved_Models/{new_fname}')
            print(f"Moved {new_fname} to saved models folder.")
    
    backups = list(zip(*[r[2] for r in results]))
    avg_preds = np.mean(np.array(backups), axis=1)

    final_errs = haversine_batch(real_coords, avg_preds)
    final_pts = [geoguessr_points(e) for e in final_errs]
    print("Final points for each image:", final_pts)
    print("Final total:", sum(final_pts))
    print("Highest points for each image:", highest_points)
    print("Highest total:", sum(highest_points))
    

    print(f"Time elapsed: {time.time()-start:.2f}s")
    return sum(final_pts), sum(highest_points), full_results, names_bad


if __name__ == "__main__":
    start_time = time.time()
    testtype = 'All' #'Validation' or 'Game' or 'Verification' or 'Super' or 'All'
    final_scores = []
    bad_models = []
    if testtype == 'All':
        for testtype in list_of_maps:
            print("\n----------------------------------------------------------------------\n")
            #print(f"Running test for {testtype}...")
            final_score, highest_score, full_results, names_bad = main(testtype)
            final_scores.append((testtype, final_score, highest_score, full_results))
            bad_models.extend(names_bad)
        print("\nFinal scores for all test types:")
        for testtype, final_score, highest_score, full_results in final_scores:
            print(f"{testtype}: {final_score}, Highest: {highest_score}, full: {full_results[0][0]}")
        
        if bad_models:
            print("\nModels that scored below average points and were not saved:")
            # count the number of times each model was not saved
            for model in set(bad_models):
                print(f"{model}: {bad_models.count(model)} times")
            
            #only save the names of models with more then the average number of times they were bad
            avg =0
            for model in set(bad_models):
                avg += bad_models.count(model)
            avg = avg / len(set(bad_models))*0.75
            avg = np.floor(avg)
            print(f"Average number of times a model was bad: {avg}")

            count =0

            bad_models = [model for model in set(bad_models) if bad_models.count(model) >= avg]
            # delete the models from the folder
            for model in bad_models:
                try:
                    os.remove(f'Roy/ML/Saved_Models_Checkpoint/Checkpoint_Models_NN/{model}')
                    #print(f"Deleted {model}")
                    count += 1
                except FileNotFoundError:
                    print(f"{model} not found, skipping deletion.")
            print(f"Deleted {count} models that scored below average points.")

                
    else:
        main(testtype)
        #main() # Uncomment this line to run the main function without any arguments and accept user input
    print(f"Execution time: {time.time() - start_time} seconds")