import re
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoImageProcessor
import s2sphere as s2
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from rich.progress import track

# ----------------------------
# CONFIG
# ----------------------------
CKPT = "google/siglip2-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = Path(r"D:/GeoGuessrGuessr/geoguesst")
TEST_DIR = Path("GeoGuessrGuessr-1/Test_Images")

LEVELS = [3, 6, 9, 12]
BEAM_SIZE = 32


# ----------------------------
# REAL LOOKUP IMPORT (already provided by you)
# ----------------------------
import importlib.util

lookup_path = Path(__file__).resolve().parents[3] / "Test_Images" / "Real_coords_lookup.py"
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
        x = self.processor(images=img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = self.model.vision_model(**x)
            feat = out.pooler_output[0]

        return F.normalize(feat, dim=0)


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
# TEST PARSING
# ----------------------------
def parse_test_image(path):
    stem = Path(path).stem
    m = re.fullmatch(r"(.+)_Test(\d+)", stem)
    if not m:
        raise ValueError(path)

    return m.group(1), int(m.group(2)) - 1


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
def save_index(index, path="s2_index.pt"):
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
    
def load_index(path="GeoGuessrGuessr-1/Roy/V2/Testing/s2_index.pt"):
    data = torch.load(path, map_location=DEVICE)

    index = S2Index()

    for level in data["maps"]:
        for k, v in data["maps"][level].items():
            index.maps[level][k] = v.to(DEVICE)

    print(f"Loaded index ← {path}")
    return index


def build_index(encoder):
    index = S2Index()

    imgs = list(TRAIN_DIR.rglob("*.png"))

    print("Indexing:", len(imgs))

    for p in track(imgs):
        try:
            lat, lon = parse_train_filename(p)
            img = Image.open(p).convert("RGB")
            emb = encoder.embed(img)

            index.add(lat, lon, emb)

        except Exception as e:
            print("skip", p, e)

    index.finalize()
    return index


# ----------------------------
# BEAM SEARCH (HIERARCHICAL PRUNING)
# ----------------------------
def beam_search(q, index: S2Index):

    beam = []

    for cid in index.all(3):
        emb = index.get(3, cid)
        if emb is None:
            continue
        beam.append((torch.dot(q, emb), cid))

    beam.sort(reverse=True)
    beam = beam[:BEAM_SIZE]

    for level in [6, 9, 12]:
        new = []

        for _, parent in beam:
            cell = s2.CellId(parent)
            children = list(cell.children(level))

            for ch in children:
                emb = index.get(level, ch.id())
                if emb is None:
                    continue

                new.append((torch.dot(q, emb), ch.id()))

        new.sort(reverse=True)
        beam = new[:BEAM_SIZE]

    return beam


# ----------------------------
# FINAL SELECTION
# ----------------------------
def final_cell(q, beam, index):
    best = None
    best_s = -1e9

    for s, cid in beam:
        emb = index.get(12, cid)
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
    q = encoder.embed(img)

    beam = beam_search(q, index)

    cid = final_cell(q, beam, index)

    cell = s2.CellId(cid)
    return cell_to_latlon(cell)


# ----------------------------
# EVALUATION (VALIDATION SET)
# ----------------------------
def evaluate(encoder, index):

    rows = []

    for p in track(sorted(TEST_DIR.glob("*"))):
        #print(p.name)

        try:
            test_type, idx = parse_test_image(p.name)
            gt = get_real_coordinates(test_type)[idx]

            img = Image.open(p).convert("RGB")

            pred = predict(img, encoder, index)

            dist = haversine(gt[0], gt[1], pred[0], pred[1])

            rows.append({
                "img": p.name,
                "dist": dist,
                "score": geoguessr_score(dist)
            })

            print(p.name, dist, geoguessr_score(dist), pred, gt)

        except Exception as e:
            print("skip", p, e)

    df = pd.DataFrame(rows)
    print(df)
    
    df_mean = df["dist"].mean()
    df_median = df["dist"].median()
    df_score_sum = df["score"].sum()

    print("\nMEAN KM:", df_mean)
    print("MEDIAN:", df_median)
    print("TOTAL SCORE:", df_score_sum)

    return df


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    print("Loading encoder...")
    encoder = SigLIPEncoder()

    print("Building S2 index from 70k images...")
    INDEX_PATH = "GeoGuessrGuessr-1/Roy/V2/Testing/s2_index.pt"

    if Path(INDEX_PATH).exists():
        print("Loading cached index...")
        index = load_index(INDEX_PATH)

    else:
        print(Path(INDEX_PATH).parent, Path(INDEX_PATH).name, Path(INDEX_PATH).exists())
        print("Building index from scratch...")
        index = build_index(encoder)

        print("Saving index...")
        save_index(index, INDEX_PATH)

    print("Running validation...")
    evaluate(encoder, index)