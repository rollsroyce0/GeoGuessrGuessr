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

# Define ocean bounding boxes and land exceptions
OCEAN_BOXES = {
    # Pacific split into four strictly-ocean quadrants (no major landmasses)
    "Pacific_NW": (0.0, 41.8, 142.08, 180.0),
    "Pacific_NW2": (41.8, 51.0, 157.0, 180.0),
    "Pacific_NE": (15.0, 48.6, -180.0, -124.7),
    "Pacific_NE2": (0.0, 15.0, -180.0, -93.0),
    "Pacific_SE": (-60.0, 0.0, -180.0, -82.0),

    # Atlantic split into precise ocean-only regions
    "Atlantic_NW": (20.0, 41.0, -71.5, -59.0),
    "Atlantic_NW2": (8.5, 46.5, -59.0, -30.0),
    "Atlantic_NW3": (46.5, 60.0, -52.0, -30.0),
    "Atlantic_NE": (29.2, 63.0, -30.0, -10.5),
    "Atlantic_NE2": (4.25, 29.2, -30.0, -18.0),
    "Atlantic_NE3": (0.0, 4.25, -30.0, 0.0),
    "Atlantic_SW": (-35.2, 0.0, -34.5, -10.0),
    "Atlantic_SE": (-35.2, 0.0, -10.0, 8.7),
    "Atlantic_S2": (-60.0, -35.2, -56.0, 136.0),

    # Indian Ocean trimmed to avoid land
    "Indian_N": (-30.0, 5.8, 51.0, 95.0),
    "Indian_S": (-35.2, -30.0, 32.0, 114.0),

    # Polar seas
    "Arctic": (70.0, 190.0, -180.0, 180.0),
    "Southern": (-190.0, -60.0, -180.0, 180.0),
}
EXCEPTIONS = {
    # Only those island groups whose boxes overlap the current OCEAN_BOXES
    "Hawaii": (18.9, 22.3, -160.3, -154.8),
    "Galápagos Islands": (-1.6, 1.667, -92.0167, -89.2667),
    "Pitcairn Islands": (-25.0667, -23.9267, -130.7372, -124.7864),
    "Northern Mariana Islands": (14.9, 18.2, 145.6, 147.1),
    "American Samoa": (-13.35, -11.0, -172.8, -169.19),
    "Bonin Islands": (24.1, 27.1, 142.0, 142.3),
    "Vanuatu": (-20.0, -13.0, 166.0, 171.0),
    "Maldives": (-0.7, 7.2, 72.5, 73.7),
    "British Indian Ocean Territory": (-7.3, -5.4, 71.3, 72.6),
    "Azores": (36.9, 39.7, -31.5, -24.4),
    "Madeira Islands": (32.4, 33.15, -17.3, -16.2),
    "Canary Islands": (27.6, 29.5, -18.3, -13.3),
    "South Georgia and the South Sandwich Islands": (-59.5, -53.0, -38.0, -26.0),
    #"Greenland": (59.0, 83.0, -74.0, -11.0),
    "Svalbard": (76.0, 81.0, 10.0, 35.0),
    #"Japan (main islands & Ogasawara)": (24.0, 46.0, 122.0, 146.0),
    "Cabo Verde": (14.8, 17.2, -25.4, -22.6),
    "Bermuda": (32.2, 32.5, -64.9, -64.5),
    "Seychelles": (-9.7, 4.6, 46.2, 55.4),
    "Mauritius": (-20.8, -19.8, 56.8, 57.9),
    "Réunion":(-21.5, -20.8, 55.0, 55.8),
}

# Margins in degrees
LON_MARGIN = 0.001
LAT_MARGIN = 0.001

def is_in_ocean(lat, lon):
    """
    Return the ocean_key if (lat, lon) is within a defined ocean box,
    excluding any exception rectangles (land islands).
    """
    # normalize lon to [-180,180]
    if lon > 180: lon -= 360
    if lon < -180: lon += 360

    for ocean, (y0, y1, x0, x1) in OCEAN_BOXES.items():
        # longitude wrap
        if x0 < x1:
            in_lon = x0 < lon < x1
        else:
            in_lon = lon > x0 or lon < x1
        if y0 < lat < y1 and in_lon:
            # check exceptions
            for ex, (ey0, ey1, ex0, ex1) in EXCEPTIONS.items():
                if ex0 < ex1:
                    in_ex_lon = ex0 < lon < ex1
                else:
                    in_ex_lon = lon > ex0 or lon < ex1
                if ey0 < lat < ey1 and in_ex_lon:
                    break
            else:
                return ocean
    return None

def snap_point(lat, lon, depth=0):
    """
    Snap (lat, lon) out of any ocean box by sampling:
      - 10 points on each edge + 4 corners of its ocean box
      - plus the 10-per-side samples + 4 corners of the three closest exception boxes
    Returns the nearest candidate that's not in any ocean box.
    """
    # helper to get box center
    def center(box):
        y0, y1, x0, x1 = box
        return ((y0+y1)/2, (x0+x1)/2)

    ocean = is_in_ocean(lat, lon)
    if ocean is None:
        return lat, lon

    # gather candidates from the ocean box
    y0, y1, x0, x1 = OCEAN_BOXES[ocean]
    candidates = []
    n_points = 100
    for i in range(n_points):
        t = i / (n_points - 1)
        # bottom, top, left, right edges
        candidates += [
            (y0 - LAT_MARGIN, x0 + t*(x1-x0)),
            (y1 + LAT_MARGIN, x0 + t*(x1-x0)),
            (y0 + t*(y1-y0), x0 - LON_MARGIN),
            (y0 + t*(y1-y0), x1 + LON_MARGIN),
        ]
    # four corners
    candidates += [
        (y0 - LAT_MARGIN, x0 - LON_MARGIN),
        (y0 - LAT_MARGIN, x1 + LON_MARGIN),
        (y1 + LAT_MARGIN, x0 - LON_MARGIN),
        (y1 + LAT_MARGIN, x1 + LON_MARGIN),
    ]

    # now include the three closest exception boxes
    # compute distances to each exception center
    ex_dists = []
    for key, (ey0, ey1, ex0, ex1) in EXCEPTIONS.items():
        cy, cx = center((ey0, ey1, ex0, ex1))
        ex_dists.append(((ey0, ey1, ex0, ex1), (cy-lat)**2 + (cx-lon)**2))
    ex_dists.sort(key=lambda e: e[1])
    n_points = int(n_points /10)
    #print(f"Exception distances: {ex_dists}")
    for box, _ in ex_dists[:3]:
        ey0, ey1, ex0, ex1 = box
        # sample its edges + corners
        
        for i in range(n_points):
            t = i / (n_points - 1)
            candidates += [
                (ey0 + LAT_MARGIN, ex0 + t*(ex1-ex0)), # bottom
                (ey1 - LAT_MARGIN, ex0 + t*(ex1-ex0)), # top
                (ey0 + t*(ey1-ey0), ex0 + LON_MARGIN), # left
                (ey0 + t*(ey1-ey0), ex1 - LON_MARGIN), # right
            ]
        candidates += [
            (ey0 + LAT_MARGIN, ex0 + LON_MARGIN), # bottom left
            (ey0 + LAT_MARGIN, ex1 - LON_MARGIN), # bottom right
            (ey1 - LAT_MARGIN, ex0 + LON_MARGIN), # top left
            (ey1 - LAT_MARGIN, ex1 - LON_MARGIN), # top right
        ]

    # sort by distance from original lat/lon
    candidates.sort(key=lambda c: (c[0]-lat)**2 + (c[1]-lon)**2)
    for ny, nx in candidates:
        if is_in_ocean(ny, nx) is None:
            return ny, nx
    return lat, lon

def snap_progress(lat, lon, depth2=0):
    path = [(lat, lon)]
    new_lat, new_lon = lat, lon
    while True:
        depth2 += 1
        #print(f"Snapping depth2 {depth2} for ({lat}, {lon})")
        ocean = is_in_ocean(new_lat, new_lon)
        if ocean is None:
            break
        new_lat, new_lon = snap_point(new_lat, new_lon)
        path.append((new_lat, new_lon))
    return path

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
    plt.legend(); plt.show(block=False); plt.pause(0.01)
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
        for i in range(len(preds)):
            preds[i,0], preds[i,1] = snap_point(preds[i,0], preds[i,1])
            #preds[i,0], preds[i,1] = snap_coords(preds[i,:])
            #preds[i,0], preds[i,1] = snap_progress(preds[i,0], preds[i,1])

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
    
    
    backups = list(zip(*[r[2] for r in results]))
    avg_preds = np.mean(np.array(backups), axis=1)

    final_errs = haversine_batch(real_coords, avg_preds)
    final_pts = [geoguessr_points(e) for e in final_errs]
    print("Final points for each image:", final_pts)
    print("Final total:", sum(final_pts))
    print("Highest points for each image:", highest_points)
    print("Highest total:", sum(highest_points))
    

    for i, path in enumerate(img_paths):
        plot_coordinates_on_map(avg_preds[i], real_coords[i], backups[i], path)

    print(f"Time elapsed: {time.time()-start:.2f}s")
    return sum(final_pts), sum(highest_points), total_points_backup


if __name__ == "__main__":
    start_time = time.time()
    testtype = 'All' #'Validation' or 'Game' or 'Verification' or 'Super' or 'All'
    final_scores = []
    if testtype == 'All':
        for testtype in ['Game', 'Validation', 'Super', 'Verification', 'Ultra', 'Extreme','Chrome', 'World', 'Task', 'Enlarged', 'Exam', 'Google']:
            print("\n----------------------------------------------------------------------\n")
            scores, highest_points, total_points_backup = main(testtype)
            final_scores.append((testtype, scores, highest_points, total_points_backup))
        print("\nFinal scores for all test types:")
        for testtype, scores, highest_points, total_points_backup in final_scores:
            print(f"{testtype}: {scores} pts")
            #print(f"Highest points: {highest_points}")
            #print(f"Total points backup: {total_points_backup}")
    else:
        main(testtype)
        #main() # Uncomment this line to run the main function without any arguments and accept user input
    print(f"Execution time: {time.time() - start_time} seconds")