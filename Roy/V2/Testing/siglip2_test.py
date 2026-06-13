from pathlib import Path
import numpy as np
import torch
from PIL import Image
from rich.progress import track
from transformers import AutoImageProcessor, SiglipVisionModel
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
#from Real_coords_lookup import get_real_coordinates
from pathlib import Path
import re

from pathlib import Path
import importlib.util

lookup_path = Path(__file__).resolve().parents[3] / "Test_Images" / "Real_coords_lookup.py"

spec = importlib.util.spec_from_file_location(
    "Real_coords_lookup",
    lookup_path
)

lookup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lookup)

get_real_coordinates = lookup.get_real_coordinates
list_test_types = lookup.list_test_types




# --------------------
# CONFIG
# --------------------
CKPT = "google/siglip2-so400m-patch14-384"

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

TRAIN_DIR = Path(r"D:/GeoGuessrGuessr/geoguesst")
CACHE_FILE = TRAIN_DIR / "siglip_cache.npz"
TEST_DIR = Path("GeoGuessrGuessr-1/Test_Images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(CKPT)
model = SiglipVisionModel.from_pretrained(CKPT).to(DEVICE).eval()

# Directory Tests
for test_type in list_test_types():
    print(f"{test_type}: {len(list(TEST_DIR.glob(f'{test_type}_*.jpg')))} images")


# --------------------
# DATA PARSING
# --------------------
def parse_coords(path):
    lat, lon = path.stem.split("_", 2)[:2]
    return float(lat), float(lon)


def get_image_list():
    return sorted([
        p for p in TRAIN_DIR.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ])


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def evaluate(
    support_paths,
    support_coords,
    support_embs
):
    rows = []

    for img_path in sorted(TEST_DIR.glob("*")):

        stem = img_path.stem

        # Example:
        # Validation_0
        # Game_3
        # TYRT_2

        test_type, idx = stem.rsplit("_", 1)
        idx = int(idx)

        true_coords = get_real_coordinates(test_type)
        true_lat, true_lon = true_coords[idx]

        pred = predict_coords(
            str(img_path),
            support_paths,
            support_coords,
            support_embs,
            k=5
        )

        dist = haversine(
            true_lat,
            true_lon,
            pred["lat"],
            pred["lon"]
        )

        score = geoguessr_score(dist)

        rows.append({
            "image": img_path.name,
            "test_type": test_type,
            "true_lat": true_lat,
            "true_lon": true_lon,
            "pred_lat": pred["lat"],
            "pred_lon": pred["lon"],
            "distance_km": dist,
            "score": score
        })

        print(
            f"{img_path.name:<25}"
            f"{dist:10.1f} km"
            f"{score:6d}"
        )

    df = pd.DataFrame(rows)

    print()
    print("Images:", len(df))
    print("Mean distance:", df.distance_km.mean())
    print("Median distance:", df.distance_km.median())
    print("Mean score:", df.score.mean())
    print("Total score:", df.score.sum())

    df.to_csv(
        "siglip_validation_results.csv",
        index=False
    )

    return df

def parse_test_image(path):
    stem = Path(path).stem

    m = re.fullmatch(r"(.+)_Test(\d+)", stem)
    if m is None:
        raise ValueError(f"Cannot parse filename: {path}")

    test_type = m.group(1)
    idx = int(m.group(2)) - 1

    return test_type, idx






def geoguessr_score(distance_km):
    return round(5000 * np.exp(-distance_km / 1492.7))

# --------------------
# EMBEDDING
# --------------------
def embed_image(path):
    img = Image.open(path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.pooler_output
    emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb[0]   # KEEP AS TORCH TENSOR (no numpy yet)


# --------------------
# CACHE
# --------------------
def load_cache():
    if not CACHE_FILE.exists():
        return {}

    data = np.load(CACHE_FILE, allow_pickle=True)

    return {
        p: e.astype(np.float32)
        for p, e in zip(data["paths"], data["embeddings"])
    }


def save_cache(cache):
    paths = np.array(list(cache.keys()), dtype=object)
    embs = np.stack([cache[p] for p in paths])

    np.savez_compressed(
        CACHE_FILE,
        paths=paths,
        embeddings=embs
    )

    print(f"Saved {len(paths)} embeddings")


# --------------------
# SUPPORT SET
# --------------------
def build_support_set():
    cache = load_cache()
    image_paths = get_image_list()

    new = 0

    for p in track(image_paths):
        key = str(p)

        if key not in cache:
            cache[key] = embed_image(p).detach().cpu().numpy()
            new += 1

    if new:
        save_cache(cache)

    paths, coords, embs = [], [], []

    for p in image_paths:
        key = str(p)
        if key not in cache:
            continue

        try:
            coords.append(parse_coords(p))
        except:
            continue

        paths.append(p)
        embs.append(cache[key])
    
    

    coords = torch.tensor(coords, device=DEVICE, dtype=torch.float32)
    embs = torch.tensor(np.stack([e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e for e in embs]), device=DEVICE, dtype=torch.float32)

    embs = torch.nn.functional.normalize(embs, dim=1)
    


    return paths, coords, embs


# --------------------
# PREDICTION
# --------------------
def predict_coords(query_image, support_paths, support_coords, support_embs, k=5):

    q = embed_image(query_image).to(DEVICE).float()
    q = torch.nn.functional.normalize(q, dim=0)

    # similarity search
    sims = support_embs @ q  # (N,)

    topk = torch.topk(sims, k)

    indices = topk.indices
    scores = topk.values

    weights = torch.softmax(scores / 0.07, dim=0)

    # IMPORTANT: support_coords already tensor on GPU → do NOT re-wrap
    pred = (support_coords[indices] * weights.unsqueeze(1)).sum(dim=0)

    neighbors = [
        (str(support_paths[i]), float(sims[i].item()))
        for i in indices.tolist()
    ]

    return {
        "lat": float(pred[0].item()),
        "lon": float(pred[1].item()),
        "neighbors": neighbors
    }


# --------------------
# MAIN
# --------------------
if __name__ == "__main__":

    print("Building support set...")
    support_paths, support_coords, support_embs = build_support_set()

    print(f"{len(support_paths)} images loaded")
    
    Average_error_km = 0
    Average_score = 0

    for img_path in sorted(TEST_DIR.glob("*")):

        try:
            test_type, idx = parse_test_image(img_path.name)
        except Exception as e:
            print(f"Skipping {img_path.name}: {e}")
            continue

        try:
            true_lat, true_lon = get_real_coordinates(test_type)[idx]
        except Exception as e:
            print(f"Lookup failed for {img_path.name}: {e}")
            continue

        pred = predict_coords(
            str(img_path),
            support_paths,
            support_coords,
            support_embs,
            k=5
        )
        dist = haversine(
            true_lat,
            true_lon,
            pred["lat"],
            pred["lon"]
        )


        score = geoguessr_score(dist)
        print(
            f"{img_path.name:<25}"
            f"{dist:10.1f} km"
            f"{score:6d}"
        )
        Average_error_km += dist
        Average_score += score
    num_images = len(list(TEST_DIR.glob("*")))
    print(f"\nAverage error: {Average_error_km / num_images:.1f} km")
    print(f"Average score: {Average_score / num_images:.1f}")
        