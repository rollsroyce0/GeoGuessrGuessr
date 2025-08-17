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

global list_of_maps
list_of_maps = ['Game',
                'Validation',
                'Super',
                'Verification',
                'Ultra',
                'Extreme',
                'Chrome',
                'World',
                'Task',
                'Enlarged',
                'Exam',
                'Google',
                'Zurich',
                'Friends',
                'Full',
                'Entire',
                'Moscow']

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
    real_coords_Zurich = np.array([[29.9590073,-95.3911924], [62.6314057,23.6289403], [34.9733313,-84.0203661], [4.3001312,117.8594655], [55.7862947,-3.9229578]])
    real_coords_Moscow = np.array([[-34.5218991,-58.5366628], [51.2135105,45.9190967], [53.1024139,-6.0640463], [37.715336,126.7597928], [47.5224219,-111.2700033]])
    real_coords_Friends = np.array([[38.9812844,-76.9781788], [59.871625,30.299387], [-1.5005364,29.621744], [59.0595843,-3.0761426], [1.7170285,103.4522982]])
    real_coords_Full = np.array([[41.102985,40.7492832], [52.5649318,-0.2828335], [47.2318584,38.8684533], [41.8301262,-70.8728116], [23.11086,72.5172045]])
    real_coords_Entire = np.array([[34.628853,136.5105634], [-22.1920816,-48.4043219], [51.073197,17.7593433], [36.575606,-79.8418298], [38.1897149,15.243788]])
    
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
    elif testtype == 'Zurich':
        real_coords = real_coords_Zurich
    elif testtype == 'Moscow':
        real_coords = real_coords_Moscow
    elif testtype == 'Friends':
        real_coords = real_coords_Friends
    elif testtype == 'Full':
        real_coords = real_coords_Full
    elif testtype == 'Entire':
        real_coords = real_coords_Entire
    else:
        raise ValueError("Invalid test type. Choose a valid one from the list.")
    
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
    print(f"Number of checkpoint models: {len(os.listdir('Roy/ML/Saved_Models_New/Checkpoint_Models_NN'))}")

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
    # Loop over all models in a second folder named 'Roy/ML/Saved_Models_New/Checkpoint_Models_NN/'
    for fname in track(sorted(os.listdir('Roy/ML/Saved_Models_New/Checkpoint_Models_NN'))):
        if 'embedding' in fname or 'lowest' in fname or not fname.endswith('.pth'):
            continue

        predictor = GeoPredictorNN().to(device).eval()
        predictor.load_state_dict(torch.load(f'Roy/ML/Saved_Models_New/Checkpoint_Models_NN/{fname}', map_location=device))

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
            #os.remove(f'Roy/ML/Saved_Models_New/Checkpoint_Models_NN/{fname}')
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
            os.rename(f'Roy/ML/Saved_Models_New/Checkpoint_Models_NN/{fname}', f'Roy/ML/Saved_Models/{new_fname}')
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
                    os.remove(f'Roy/ML/Saved_Models_New/Checkpoint_Models_NN/{model}')
                    #print(f"Deleted {model}")
                    count += 1
                except FileNotFoundError:
                    print(f"{model} not found, skipping deletion.")
            print(f"Deleted {count} models that scored below average points.")

                
    else:
        main(testtype)
        #main() # Uncomment this line to run the main function without any arguments and accept user input
    print(f"Execution time: {time.time() - start_time} seconds")