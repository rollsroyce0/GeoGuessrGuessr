import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoImageProcessor
import s2sphere as s2
import pandas as pd
import argparse
from math import radians, sin, cos, sqrt, atan2
from rich.progress import track
import matplotlib.pyplot as plt
from Roy.Helper_Functions.project_utils import (
    get_s2_index_path,
    get_test_image_path,
    get_test_images_dir,
    parse_test_image as parse_test_image_name,
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='ProtoNet GeoGuessr Test')
parser.add_argument('--debug', action='store_true', help='Enable extensive debug statements')
parser.add_argument('--gui', action='store_true', help='Enable GUI mode with loading bar and interactive results')
parser.add_argument('--hist', action='store_true', help='Show histogram of validation distances')
args = parser.parse_args()

# Debug flag
DEBUG = args.debug

def debug_print(*messages):
    """Print debug messages only when debug flag is enabled"""
    if DEBUG:
        print("[DEBUG]", *messages)

# ----------------------------
# CONFIG
# ----------------------------
CKPT = "google/siglip2-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = Path(r"D:/GeoGuessrGuessr/geoguesst")
TEST_DIR = get_test_images_dir()

debug_print(f"Using device: {DEVICE}")
debug_print(f"Train directory: {TRAIN_DIR}")
debug_print(f"Test directory: {TEST_DIR}")

LEVELS = [3, 7, 11, 15]
BEAM_SIZE = 32


# ----------------------------
# REAL LOOKUP IMPORT (already provided by you)
# ----------------------------
import importlib.util

lookup_path = get_test_images_dir() / "Real_coords_lookup.py"
spec = importlib.util.spec_from_file_location("lookup", lookup_path)
lookup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lookup)

get_real_coordinates = lookup.get_real_coordinates


# ----------------------------
# SIGLIP ENCODER (FIXED)
# ----------------------------
class SigLIPEncoder:
    def __init__(self):
        self.model = AutoModel.from_pretrained(CKPT).to(DEVICE).eval()
        self.processor = AutoImageProcessor.from_pretrained(CKPT)

        dummy = Image.new("RGB", (384, 384))
        x = self.processor(images=dummy, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = self.model.vision_model(**x)
            self.dim = out.pooler_output.shape[-1]

    def embed(self, img):
        debug_print("Embedding image with SigLIP encoder")
        x = self.processor(images=img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = self.model.vision_model(**x)
            feat = out.pooler_output[0]

        debug_print(f"Feature shape before normalization: {feat.shape}")
        result = F.normalize(feat, dim=0)
        debug_print(f"Feature shape after normalization: {result.shape}")
        return result


# ----------------------------
# TRAIN IMAGE PARSING
# ----------------------------
def parse_train_filename(path: Path):
    """
    0.33659_6.7055088_Index__Index_2.png
    """
    parts = path.stem.split("_")

    lat = float(parts[0])
    lon = float(parts[1])

    return lat, lon


# ----------------------------
# S2 UTILITIES
# ----------------------------
def latlon_to_cell(lat, lon, level):
    ll = s2.LatLng.from_degrees(lat, lon)
    return s2.CellId.from_lat_lng(ll).parent(level)


def cell_to_latlon(cell):
    ll = cell.to_lat_lng()
    return ll.lat().degrees, ll.lng().degrees


# ----------------------------
# GEO METRICS
# ----------------------------
def haversine(a, b, c, d):
    R = 6371.0
    a, b, c, d = map(radians, [a, b, c, d])
    da = c - a
    db = d - b
    x = sin(da/2)**2 + cos(a)*cos(c)*sin(db/2)**2
    return 2 * R * atan2(sqrt(x), sqrt(1-x))


def geoguessr_score(d):
    return round(5000 * np.exp(-d / 1492.7))


# ----------------------------
# S2 INDEX (PROTOTYPES)
# ----------------------------
class S2Index:
    def __init__(self):
        self.maps = {l: {} for l in LEVELS}

    def add(self, lat, lon, emb):
        cell = s2.CellId.from_lat_lng(
            s2.LatLng.from_degrees(lat, lon)
        )

        for l in LEVELS:
            cid = cell.parent(l).id()

            if cid not in self.maps[l]:
                self.maps[l][cid] = []

            self.maps[l][cid].append(emb)

    def finalize(self):
        for l in LEVELS:
            for k in self.maps[l]:
                v = torch.stack(self.maps[l][k]).mean(0)
                self.maps[l][k] = F.normalize(v, dim=0)

    def get(self, level, cid):
        return self.maps[level].get(cid, None)

    def all(self, level):
        return list(self.maps[level].keys())


# ----------------------------
# BUILD INDEX FROM 70K IMAGES
# ----------------------------
def save_index(index, path=None):
    if path is None:
        path = get_s2_index_path()
    data = {
        "maps": {}
    }

    for level in index.maps:
        data["maps"][level] = {
            k: v.detach().cpu()
            for k, v in index.maps[level].items()
        }

    torch.save(data, path)
    print(f"Saved index → {path}")
    
def load_index(path=None):
    if path is None:
        path = get_s2_index_path()
    data = torch.load(path, map_location=DEVICE)

    index = S2Index()

    # Check if the LEVELS in the loaded data match the expected LEVELS, else build a new index and save it
    loaded_levels = set(data["maps"].keys())
    if loaded_levels != set(LEVELS):
        print(f"Loaded levels {loaded_levels} do not match expected levels {set(LEVELS)}")
        return None

    for level in data["maps"]:
        for k, v in data["maps"][level].items():
            index.maps[level][k] = v.to(DEVICE)

    print(f"Loaded index from {path}")
    return index


def build_index(encoder):
    debug_print("Starting to build S2 index")
    index = S2Index()

    imgs = list(TRAIN_DIR.rglob("*.png"))
    debug_print(f"Found {len(imgs)} images to index")

    print("Indexing:", len(imgs))

    for p in track(imgs):
        try:
            lat, lon = parse_train_filename(p)
            debug_print(f"Processing image: {p.name} at ({lat:.6f}, {lon:.6f})")
            img = Image.open(p).convert("RGB")
            emb = encoder.embed(img)
            debug_print(f"Embedding created for {p.name}")

            index.add(lat, lon, emb)

        except Exception as e:
            debug_print(f"Error processing {p}: {e}")
            print("skip", p, e)

    debug_print("Finalizing index...")
    index.finalize()
    debug_print("Index building completed")
    return index


# ----------------------------
# BEAM SEARCH (HIERARCHICAL PRUNING)
# ----------------------------

# Automatically adjust the beam size based on the number of candidates at each level. Reads the LEVELS constant to determine how many levels there are and prunes the beam accordingly. This allows for a more flexible search that can adapt to different index sizes and distributions.
def beam_search(q, index: S2Index):
    debug_print("Starting beam search")
    debug_print(f"Query vector shape: {q.shape}")

    beam = []
    # Parametric
    for cid in index.all(LEVELS[0]):
        emb = index.get(LEVELS[0], cid)
        if emb is None:
            continue
        similarity = torch.dot(q, emb)
        beam.append((similarity, cid))

    debug_print(f"Initial beam size at level {LEVELS[0]}: {len(beam)}")
    beam.sort(reverse=True)
    beam = beam[:BEAM_SIZE]
    debug_print(f"Beam after pruning at level {LEVELS[0]}: {len(beam)} items")

    for level in LEVELS[1:]:
        new = []
        debug_print(f"Processing level {level}")

        for _, parent in beam:
            cell = s2.CellId(parent)
            children = list(cell.children(level))

            for ch in children:
                emb = index.get(level, ch.id())
                if emb is None:
                    continue

                similarity = torch.dot(q, emb)
                new.append((similarity, ch.id()))

        new.sort(reverse=True)
        beam = new[:BEAM_SIZE]
        debug_print(f"Beam after pruning at level {level}: {len(beam)} items")

    debug_print("Beam search completed")
    return beam


# ----------------------------
# FINAL SELECTION
# ----------------------------
def final_cell(q, beam, index):
    best = None
    best_s = -1e9

    for s, cid in beam:
        emb = index.get(LEVELS[-1], cid)
        if emb is None:
            continue

        v = torch.dot(q, emb)

        if v > best_s:
            best_s = v
            best = cid

    return best


# ----------------------------
# PREDICT
# ----------------------------
def predict(img, encoder, index):
    debug_print("Starting prediction")
    q = encoder.embed(img)

    beam = beam_search(q, index)
    debug_print(f"Beam search returned {len(beam)} candidates")

    cid = final_cell(q, beam, index)
    debug_print(f"Final cell ID: {cid}")

    cell = s2.CellId(cid)
    pred_lat, pred_lon = cell_to_latlon(cell)
    debug_print(f"Predicted coordinates: lat={pred_lat:.6f}, lon={pred_lon:.6f}")

    return pred_lat, pred_lon


# ----------------------------
# HISTOGRAM PLOTTING
# ----------------------------
def plot_histogram(distances):
    """Plot histogram of validation distances"""
    debug_print("Plotting histogram of distances")

    # Calculate statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)

    # Create figure with two subplots
    plt.figure(figsize=(18, 8))

    # First histogram: Full distribution with 50 bins
    plt.subplot(1, 2, 1)
    n1, bins1, patches1 = plt.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='blue')

    plt.title('Full Distribution of Validation Distances (50 bins)', fontsize=14)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add mean and median lines
    plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.1f} km')
    plt.axvline(median_dist, color='green', linestyle='--', linewidth=2, label=f'Median: {median_dist:.1f} km')
    plt.legend(fontsize=10)

    # Second histogram: Upper 50% of scores with 25 bins
    plt.subplot(1, 2, 2)

    # Calculate threshold for upper 50% (lower distances = better scores)
    upper_50_threshold = np.percentile(distances, 50)
    upper_50_distances = distances[distances <= upper_50_threshold]

    n2, bins2, patches2 = plt.hist(upper_50_distances, bins=25, edgecolor='black', alpha=0.7, color='orange')

    plt.title('Upper 50% of Scores (25 bins)', fontsize=14)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add mean and median lines for the upper 50% data
    upper_50_mean = np.mean(upper_50_distances)
    upper_50_median = np.median(upper_50_distances)

    plt.axvline(upper_50_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {upper_50_mean:.1f} km')
    plt.axvline(upper_50_median, color='green', linestyle='--', linewidth=2, label=f'Median: {upper_50_median:.1f} km')
    plt.legend(fontsize=10)

    plt.suptitle('Validation Distance Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# ----------------------------
# EVALUATION (VALIDATION SET)
# ----------------------------
def evaluate(encoder, index, progress_callback=None):
    debug_print("Starting evaluation")

    rows = []
    processed_images = 0
    total_images = len(list(TEST_DIR.glob("*")))

    # Get all test images for progress tracking
    test_images = sorted(TEST_DIR.glob("*"))

    for i, p in enumerate(track(test_images)):
        debug_print(f"\nProcessing image: {p.name}")

        try:
            test_type, idx = parse_test_image_name(p.name)
            gt = get_real_coordinates(test_type)[idx]
            debug_print(f"  - Ground truth: lat={gt[0]:.6f}, lon={gt[1]:.6f}")

            img = Image.open(p).convert("RGB")

            pred = predict(img, encoder, index)
            debug_print(f"  - Prediction: lat={pred[0]:.6f}, lon={pred[1]:.6f}")

            dist = haversine(gt[0], gt[1], pred[0], pred[1])
            debug_print(f"  - Distance: {dist:.2f} km")

            score = geoguessr_score(dist)
            debug_print(f"  - Score: {score}")

            # Store all needed data including coordinates
            rows.append({
                "img": p.name,
                "dist": dist,
                "score": score,
                "pred_lat": pred[0],
                "pred_lon": pred[1],
                "real_lat": gt[0],
                "real_lon": gt[1],
                "test_type": test_type,
                "image_idx": idx
            })

            debug_print(p.name, dist, geoguessr_score(dist), pred, gt)

            # Update progress callback if provided
            if progress_callback:
                progress = (i + 1) / total_images * 100
                progress_callback(progress, f"Processing {p.name} ({i+1}/{total_images})")

            processed_images += 1

        except Exception as e:
            debug_print(f"Error processing {p.name}: {e}")
            print("skip", p, e)

    df = pd.DataFrame(rows)
    print(df.sort_values("dist").head(10))

    df_mean = df["dist"].mean()
    df_median = df["dist"].median()
    df_score_sum = df["score"].sum()

    print("\nMEAN KM:", df_mean)
    print("MEDIAN:", df_median)
    #print("TOTAL SCORE:", df_score_sum)
    print("TOTAL IMAGES PROCESSED:", processed_images)
    print("AVERAGE SCORE PER IMAGE:", df_score_sum / processed_images if processed_images > 0 else 0)

    debug_print(f"Evaluation completed. Processed {processed_images} images.")
    debug_print(f"Mean distance: {df_mean:.2f} km")
    debug_print(f"Median distance: {df_median:.2f} km")
    debug_print(f"Total score: {df_score_sum}")
    debug_print(f"Average score per image: {df_score_sum / processed_images if processed_images > 0 else 0:.2f}")

    return df


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    debug_print("Starting ProtoNet test script")

    # Check if GUI mode is enabled
    GUI_MODE = args.gui

    if GUI_MODE:
        debug_print("GUI mode enabled")
        print("Starting ProtoNet test with GUI...")

        # Import and run GUI
        try:
            from Roy.V2.GUI.gui_main import run_gui
            run_gui()
        except ImportError as e:
            print(f"Failed to import GUI module: {e}")
            print("Falling back to console mode...")
            GUI_MODE = False
        except Exception as e:
            print(f"GUI error: {e}")
            print("Falling back to console mode...")
            GUI_MODE = False

    if not GUI_MODE:
        debug_print("Running in console mode")

        print("Loading encoder...")
        encoder = SigLIPEncoder()

        print("Building S2 index from 70k images...")
        INDEX_PATH = get_s2_index_path()

        if Path(INDEX_PATH).exists():
            print("Loading cached index...")
            index = load_index(INDEX_PATH)
            if index is None:
                print("Cached index is invalid, rebuilding...")
                index = build_index(encoder)
                save_index(index, INDEX_PATH)
            debug_print(f"Loaded index from cache: {INDEX_PATH}")
        else:
            debug_print(f"Index file not found at {INDEX_PATH}, building from scratch")
            print(Path(INDEX_PATH).parent, Path(INDEX_PATH).name, Path(INDEX_PATH).exists())
            print("Building index from scratch...")
            index = build_index(encoder)

            print("Saving index...")
            save_index(index, INDEX_PATH)
            debug_print(f"Index saved to {INDEX_PATH}")

        print("Running validation...")
        debug_print("Starting validation process")
        df = evaluate(encoder, index)

        # Plot histogram if --hist flag is set
        if args.hist:
            debug_print("Histogram flag detected, plotting distances")
            distances = df["dist"].values
            plot_histogram(distances)

        debug_print("Script completed successfully")
