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

    # Load and preprocess images once
    if testtype is None:
        testtype = input("Enter test type (Game, Validation, Super, Verification, Ultra, Extreme, Chrome): ")
    if testtype not in list_of_maps:
        raise ValueError("Invalid test type. Choose 'Game', 'Validation', 'Super', 'Ultra', or any other.")
    images, img_paths = load_images('Roy/Test_Images', testtype)
    images = images.to(device)

    # Real coordinates
    # Initialize embedding model
    real_coords = get_real_coordinates(testtype)
    
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

    # Loop over predictor weights
    for fname in sorted(os.listdir('Roy/ML/Saved_Models')):
        if 'embedding' in fname or 'lowest' in fname or not fname.endswith('.pth'):
            continue
        #print(f"Evaluating model: {fname}")
        predictor = GeoPredictorNN().to(device).eval()
        predictor.load_state_dict(torch.load(f'Roy/ML/Saved_Models/{fname}', map_location=device))

        with torch.no_grad():
            preds = predictor(embeddings.to(device)).cpu().numpy()
            preds[:,0] = (preds[:,0]+90)%180 - 90
            preds[:,1] = (preds[:,1]+180)%360 - 180

        errs = haversine_batch(real_coords, preds)
        pts = [geoguessr_points(e) for e in errs]
        total_pts = sum(pts)
        if total_pts <0 or np.isnan(total_pts) or total_pts > 25000:
            print(f"Skipping {fname} due to invalid total points: {total_pts}")
            continue
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
    print(f"Top 10 models for {testtype}:")
    for i, (fname, total_pts, preds) in enumerate(results):
        print(f"{i+1}: {fname} - {total_pts} pts")
        #print(preds)
    # Save the testtype and the best three models to a file
    # Check if the file exists, if not create it
    if not os.path.exists(f'Roy/Test_Images/Best_models_{testtype}_check.txt'):
        # Throw an error if the file does not exist
        open(f'Roy/Test_Images/Best_models_{testtype}_check.txt', 'x')
    # remove all text from the file
    with open(f'Roy/Test_Images/Best_models_{testtype}_check.txt', 'r+') as f:
        #one=1
        # remove everything from the file
        f.truncate(0)

    
    with open(f'Roy/Test_Images/Best_models_{testtype}_check.txt', 'a') as f:
        #one=1
        # remove everything from the file
        
        f.write("Best 10 models for each test type:\n")
        f.write(f"{testtype}: {results[0][0]}, {results[1][0]}, {results[2][0]}, {results[3][0]}, {results[4][0]}, {results[5][0]}, {results[6][0]}, {results[7][0]}, {results[8][0]}, {results[9][0]}\n")
    
    backups = list(zip(*[r[2] for r in results]))
    avg_preds = np.mean(np.array(backups), axis=1)
    #print(results)

    final_errs = haversine_batch(real_coords, avg_preds)
    #print(len(final_errs), len(real_coords))
    final_pts = [geoguessr_points(e) for e in final_errs]
    print("Final points for each image:", final_pts)
    print("Final total:", sum(final_pts))
    print("Highest points for each image:", highest_points)
    print("Highest total:", sum(highest_points))
    
    # Calculate a Difficulty score for each image based on the standard deviation of the predictions
    if len(errors) >=10:
        errors = np.sort(errors, axis=0)[:-10] # Remove the 10 highest errors or each individual image disregarding model order
    else:
        #combine the errors and low_errors
        errors = np.concatenate((errors, low_errors), axis=0)
        errors = np.sort(errors, axis=0)[:-10] # Remove the 10 highest errors or each individual image disregarding model order
    errors = np.array(errors)
    points_backup = np.sort(points_backup, axis=0)[-25:]
    difficulty_scores = np.std(errors, axis=0) + 0.4*np.mean(errors, axis=0) # Add the mean to the std to get a more accurate score
    #print("Errors:", errors)
    print("Difficulty scores raw:", difficulty_scores)
    # Normalize these on a scale 0-10, where an std dev of 2500 would be a difficulty of 10 and 0 would be 0. However this is not a linear scale, so we will use a logarithmic scale.
    # We will use a base of 10, so that 10^0 = 1 and 10^1 = 10. This means that a difficulty of 0 would be 0 and a difficulty of 10 would be 10.
    # We will also use a minimum difficulty of 1, so that we don't get negative scores.
    # A Difficulty of 10 means the std is 4500km or above
    
    
    difficulty_scores = np.log10(difficulty_scores/1000 + 1) * 8.502741537 # Max difficulty is now 1000
    #difficulty_scores = np.clip(difficulty_scores, 0, 10)
    difficulty_scores = difficulty_scores**3
    difficulty_scores = np.round(difficulty_scores, 3)
    print("Difficulty scores for each image:", difficulty_scores)
    print("Average difficulty score of this round:", np.round(np.mean(difficulty_scores), 3))
    # add the average difficutly score for the test type to a file

    with open(f'Roy/Test_Images/Difficulty_scores_check.txt', 'a') as f:
        f.write(f"{testtype}: {np.round(np.mean(difficulty_scores), 3)}, Highest: {np.round(np.max(difficulty_scores), 3)}, Lowest: {np.round(np.min(difficulty_scores), 3)}\n")
        
        # remove any duplicate lines (it is a duplicate, if the first 5 characters are the same)
    with open(f'Roy/Test_Images/Difficulty_scores_check.txt', 'r') as f:
        lines = f.readlines()
    
    # remove duplicates by checking the first 5 characters of each line
    seen = set()
    lines = [line for line in reversed(lines) if not (line[:5] in seen or seen.add(line[:5]))]
    lines = reversed(lines)  # reverse the lines back to original order
    # write the lines back to the file
    
    # sort the lines by the difficulty score (the second value in the line)
    lines = sorted(lines, key=lambda x: float(x.split(':')[1].split(',')[0]), reverse=True)
    
    with open(f'Roy/Test_Images/Difficulty_scores_check.txt', 'w') as f:
        f.writelines(lines)
    # Calculate average and median scores
    total_points_backup = np.array(total_points_backup)
    total_points_backup = np.sort(total_points_backup, axis=0)[-25:]  # Keep the top 25 scores
    
    avg_scores = np.mean(total_points_backup, axis=0)
    median_scores = np.median(total_points_backup, axis=0)

    print(f"Time elapsed: {time.time()-start:.2f}s")
    return sum(final_pts), sum(highest_points), np.round(np.mean(difficulty_scores), 3), avg_scores, median_scores, avg_preds, real_coords, final_errs, final_pts, img_paths, highest_points



if __name__ == "__main__":
    start_time = time.time()
    testtype = 'All' #'Validation' or 'Game' or 'Verification' or 'Super' or 'All'
    errors = []
    final_scores = []
    if testtype == 'All':
        for testtype in list_of_maps:
            print("\n----------------------------------------------------------------------\n")
            #print(f"Running test for {testtype}...")
            final_score, highest_score, difficulty_score, avg_scores, median_scores, avg_preds, real_coords, final_errs, final_pts, img_paths, highest_points = main(testtype)
            errors.extend(final_errs)
            final_scores.append((testtype, final_score, highest_score, difficulty_score, avg_scores, median_scores, highest_points, final_pts, img_paths))
        print("\nFinal scores for all test types:")
        for testtype, final_score, highest_score, difficulty_score, avg_scores, median_scores, highest_points, final_pts, img_paths in final_scores:
            print(f"{testtype}: {final_score}, Highest: {highest_score}, Avg of Difficulty: {difficulty_score}, Avg Scores: {avg_scores}, Median Scores: {median_scores}, Highest Points: {highest_points}")
        #generate a best set of 5 images with the highest points for each image across all test types
        pointies = np.array([fs[7] for fs in final_scores])
        #print('pointies:', pointies)
        #print('pointies shape:', pointies.shape)
        # Get the indices of the top 5 highest points for each image, preserving test_type
        top_5_indices = np.argsort(pointies, axis=0)[-5:]
        # also print the sum of the total top 5 scores disregarding test type or image
        p = pointies.flatten()
        p = np.sort(p)[-5:]
        print(f"\nSum of overall top 5 scores disregarding test type or image: {np.sum(p)}")
        
        # For each image, print the test_type and score of the top 5
        for img_idx in range(pointies.shape[1]):
            print(f"\nImage {img_idx+1} top 5 scores and test types:")
            for idx in reversed(top_5_indices[:, img_idx]):
                print(f"  {final_scores[idx][0]}: {pointies[idx, img_idx]}")

        avg_avg_scores = np.mean([fs[4] for fs in final_scores], axis=0)
        avg_median_scores = np.mean([fs[5] for fs in final_scores], axis=0)
        avg_highest_points = np.mean([fs[6] for fs in final_scores], axis=0)
        print(f"\nAverage scores across all test types:\nAvg Scores: {avg_avg_scores}, Median Scores: {avg_median_scores}, Highest Points: {avg_highest_points}")
        print(f"\nOverall average error across all test types: {np.mean(errors)} km, Median error: {np.median(errors)} km")
            
    else:
        final_score, highest_score, difficulty_score, avg_scores, median_scores, avg_preds, real_coords, final_errs, final_pts, img_paths, highest_points = main(testtype)
        print(f"\nFinal score for {testtype}: {final_score}, Highest: {highest_score}, Avg of Difficulty: {difficulty_score}, Avg Scores: {avg_scores}, Median Scores: {median_scores}, Highest Points: {highest_points}")
        #main() # Uncomment this line to run the main function without any arguments and accept user input
        
    print(f"Execution time: {time.time() - start_time} seconds")