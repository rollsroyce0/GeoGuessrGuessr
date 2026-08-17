import argparse
import importlib.util
import os
import sys
import time
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s2sphere as s2
import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import track
from transformers import AutoImageProcessor, AutoModel

# Add project root to Python path for Roy imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Roy.Helper_Functions.project_utils import (
    get_s2_index_path,
    get_test_image_path,
    get_test_images_dir,
    parse_test_image as parse_test_image_name,
)

DEBUG = False


def debug_print(*messages):
    if DEBUG:
        print("[DEBUG]", *messages)


def configure_debug(enabled: bool):
    global DEBUG
    DEBUG = bool(enabled)


# ----------------------------
# CONFIG
# ----------------------------
CKPT = "google/siglip2-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = Path(r"D:/GeoGuessrGuessr/geoguesst")
TEST_DIR = get_test_images_dir()
EMBED_CACHE_PATH = get_s2_index_path().with_name("s2_embeddings_cache_v2.pt")
CACHE_VERSION = 2
TRAIN_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

LEVELS = [3, 7, 11, 15]
BEAM_SIZE = 100
TOP_K_REFINEMENT = 10
LEVEL_WEIGHTS = {3: 0.12, 7: 0.20, 11: 0.28, 15: 0.40}


# ----------------------------
# REAL LOOKUP IMPORT
# ----------------------------
_LOOKUP_MODULE = None


def _load_lookup_module():
    global _LOOKUP_MODULE

    if _LOOKUP_MODULE is not None:
        return _LOOKUP_MODULE

    lookup_path = get_test_images_dir() / "Real_coords_lookup.py"
    spec = importlib.util.spec_from_file_location("lookup", lookup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load real coordinate lookup from {lookup_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOOKUP_MODULE = module
    return _LOOKUP_MODULE


def get_real_coordinates(test_type):
    module = _load_lookup_module()
    return module.get_real_coordinates(test_type)


# ----------------------------
# SIGLIP ENCODER
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

    def _prepare_images(self, images):
        prepared = []
        for img in images:
            if hasattr(img, "convert"):
                prepared.append(img.convert("RGB"))
            elif isinstance(img, (str, os.PathLike)):
                with Image.open(img) as opened:
                    prepared.append(opened.convert("RGB"))
            else:
                raise TypeError(f"Unsupported image object: {type(img)!r}")
        return prepared

    def embed(self, img):
        return self.embed_batch([img])[0]

    def embed_batch(self, images):
        if not images:
            return torch.empty((0, self.dim), device=DEVICE, dtype=torch.float32)

        prepared = self._prepare_images(images)
        inputs = self.processor(images=prepared, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = self.model.vision_model(**inputs)
            feats = out.pooler_output

        feats = F.normalize(feats, dim=-1)
        debug_print(f"Embedded batch shape: {feats.shape}")
        return feats


# ----------------------------
# TRAIN IMAGE PARSING
# ----------------------------
def parse_train_filename(path: Path):
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {path.name}")
    lat = float(parts[0])
    lon = float(parts[1])
    return lat, lon


def get_train_images():
    return sorted(
        p for p in TRAIN_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in TRAIN_IMAGE_SUFFIXES
    )


def train_cache_key(path: Path):
    return str(path.relative_to(TRAIN_DIR))


def file_signature(path: Path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def load_embedding_cache(path=None):
    if path is None:
        path = EMBED_CACHE_PATH

    if not path.exists():
        return None

    data = torch.load(path, map_location="cpu")
    if data.get("version") != CACHE_VERSION:
        return None
    if data.get("train_dir") != str(TRAIN_DIR):
        return None
    if data.get("levels") != LEVELS:
        return None
    return data


def save_embedding_cache(data, path=None):
    if path is None:
        path = EMBED_CACHE_PATH

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"Saved embedding cache → {path}")


# ----------------------------
# S2 UTILITIES
# ----------------------------
def latlon_to_cell(lat, lon, level):
    ll = s2.LatLng.from_degrees(lat, lon)
    return s2.CellId.from_lat_lng(ll).parent(level)


def cell_to_latlon(cell):
    ll = cell.to_lat_lng()
    return ll.lat().degrees, ll.lng().degrees


def latlon_to_unit_vector(lat, lon):
    lat_r = radians(lat)
    lon_r = radians(lon)
    return np.array(
        [
            cos(lat_r) * cos(lon_r),
            cos(lat_r) * sin(lon_r),
            sin(lat_r),
        ],
        dtype=np.float64,
    )


def unit_vector_to_latlon(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot convert zero vector to lat/lon")

    x, y, z = vec / norm
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return float(lat), float(lon)


# ----------------------------
# GEO METRICS
# ----------------------------
def haversine(a, b, c, d):
    R = 6371.0
    a, b, c, d = map(radians, [a, b, c, d])
    da = c - a
    db = d - b
    x = sin(da / 2) ** 2 + cos(a) * cos(c) * sin(db / 2) ** 2
    return 2 * R * atan2(sqrt(x), sqrt(1 - x))


def geoguessr_score(d):
    return round(5000 * np.exp(-d / 1492.7))


# ----------------------------
# S2 INDEX
# ----------------------------
class S2Index:
    def __init__(self):
        self.maps = {level: {} for level in LEVELS}

    def add(self, lat, lon, emb):
        cell = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lon))

        for level in LEVELS:
            cid = cell.parent(level).id()
            if cid not in self.maps[level]:
                self.maps[level][cid] = {"sum": emb.detach().clone(), "count": 1}
                continue

            self.maps[level][cid]["sum"] += emb
            self.maps[level][cid]["count"] += 1

    def finalize(self):
        for level in LEVELS:
            for cid in self.maps[level]:
                entry = self.maps[level][cid]
                avg = entry["sum"] / entry["count"]
                self.maps[level][cid] = F.normalize(avg, dim=0)

    def get(self, level, cid):
        return self.maps[level].get(cid, None)

    def all(self, level):
        return list(self.maps[level].keys())

    def iter_level(self, level):
        return self.maps[level].items()


# ----------------------------
# BUILD INDEX FROM TRAIN IMAGES
# ----------------------------
def save_index(index, path=None):
    if path is None:
        path = get_s2_index_path()

    data = {"version": 1, "levels": LEVELS, "maps": {}}
    for level in index.maps:
        data["maps"][level] = {k: v.detach().cpu() for k, v in index.maps[level].items()}

    torch.save(data, path)
    print(f"Saved index → {path}")


def load_index(path=None):
    if path is None:
        path = get_s2_index_path()

    data = torch.load(path, map_location=DEVICE)
    index = S2Index()

    if "version" in data and data["version"] != 1:
        print(f"Cached index version {data['version']} is not supported")
        return None

    loaded_levels = set(data.get("levels", data["maps"].keys()))
    if loaded_levels != set(LEVELS):
        print(f"Loaded levels {loaded_levels} do not match expected levels {set(LEVELS)}")
        return None

    for level in data["maps"]:
        for k, v in data["maps"][level].items():
            index.maps[level][k] = v.to(DEVICE)

    print(f"Loaded index from {path}")
    return index


def build_index(encoder, batch_size=16):
    debug_print("Starting index build")
    index = S2Index()
    imgs = get_train_images()
    debug_print(f"Images found: {len(imgs)}")

    print("Indexing:", len(imgs))

    current_files = [train_cache_key(p) for p in imgs]
    current_signatures = torch.tensor([file_signature(p) for p in imgs], dtype=torch.long)

    cache = load_embedding_cache()
    if cache is not None and cache.get("files") == current_files and torch.equal(cache.get("signatures"), current_signatures):
        debug_print("Using cached training embeddings")
        cached_coords = cache["coords"]
        cached_embs = cache["embeddings"]

        for coord, emb in zip(cached_coords, cached_embs):
            index.add(
                float(coord[0].item()),
                float(coord[1].item()),
                emb.to(DEVICE, dtype=torch.float32),
            )
    else:
        debug_print("Building a fresh embedding cache")
        coords = []
        embeddings = []
        processed = 0
        batch = []
        batch_meta = []

        def flush_batch():
            nonlocal processed
            if not batch:
                return

            batch_embs = encoder.embed_batch(batch).detach().cpu()
            for (lat, lon, path), emb in zip(batch_meta, batch_embs):
                debug_print(f"Embedding added: {path.name}")
                index.add(lat, lon, emb.to(DEVICE, dtype=torch.float32))
                coords.append((lat, lon))
                embeddings.append(emb.to(dtype=torch.float16))
                processed += 1

            batch.clear()
            batch_meta.clear()

        for p in track(imgs):
            try:
                lat, lon = parse_train_filename(p)
                batch.append(p)
                batch_meta.append((lat, lon, p))

                if len(batch) >= batch_size:
                    flush_batch()
            except Exception as e:
                debug_print(f"Error processing {p}: {e}")
                print("skip", p, e)

        flush_batch()

        if coords and processed == len(imgs):
            save_embedding_cache({
                "version": CACHE_VERSION,
                "train_dir": str(TRAIN_DIR),
                "levels": LEVELS,
                "files": current_files,
                "signatures": current_signatures,
                "coords": torch.tensor(coords, dtype=torch.float32),
                "embeddings": torch.stack(embeddings),
            })
        elif processed != len(imgs):
            print("Embedding cache not saved because some training images failed to process.")

    index.finalize()
    debug_print("Index build completed")
    return index


# ----------------------------
# BETTER BEAM SEARCH
# ----------------------------
def expand_to_level(parent_id, target_level):
    cells = [s2.CellId(parent_id)]
    while cells and cells[0].level() < target_level:
        next_cells = []
        for cell in cells:
            next_cells.extend(list(cell.children()))
        cells = next_cells
    return cells


def beam_search(q, index: S2Index, beam_size=BEAM_SIZE):
    debug_print("Starting beam search")

    q = F.normalize(q, dim=0)
    beam = []
    for cid, emb in index.iter_level(LEVELS[0]):
        if emb is None:
            continue
        similarity = float(torch.dot(q, emb))
        beam.append((similarity * LEVEL_WEIGHTS[LEVELS[0]], similarity, cid, LEVELS[0]))

    beam = heapq.nlargest(beam_size, beam, key=lambda item: item[0])

    for level in LEVELS[1:]:
        next_candidates = {}
        for path_score, _, parent_cid, _ in beam:
            children = expand_to_level(parent_cid, level)
            for child in children:
                emb = index.get(level, child.id())
                if emb is None:
                    continue
                similarity = float(torch.dot(q, emb))
                total_score = path_score + similarity * LEVEL_WEIGHTS[level]
                current = next_candidates.get(child.id())
                if current is None or total_score > current[0]:
                    next_candidates[child.id()] = (total_score, similarity, child.id(), level)

        if not next_candidates:
            break

        beam = heapq.nlargest(beam_size, next_candidates.values(), key=lambda item: item[0])

    debug_print(f"Beam search finished with {len(beam)} candidates")
    return beam


# ----------------------------
# FINAL PREDICTION
# ----------------------------
def final_prediction(beam, top_k=TOP_K_REFINEMENT):
    if not beam:
        return None, None, None

    ranked = sorted(beam, key=lambda item: item[0], reverse=True)
    top = ranked[: min(top_k, len(ranked))]
    scores = torch.tensor([item[0] for item in top], dtype=torch.float32)
    weights = torch.softmax(scores, dim=0).tolist()

    best_cid = top[0][2]
    best_lat, best_lon = cell_to_latlon(s2.CellId(best_cid))

    vector = np.zeros(3, dtype=np.float64)
    for weight, (_, _, cid, _) in zip(weights, top):
        lat, lon = cell_to_latlon(s2.CellId(cid))
        vector += weight * latlon_to_unit_vector(lat, lon)

    if np.linalg.norm(vector) < 1e-12:
        pred_lat, pred_lon = best_lat, best_lon
    else:
        pred_lat, pred_lon = unit_vector_to_latlon(vector)

    return best_cid, pred_lat, pred_lon


# ----------------------------
# PREDICT
# ----------------------------
def predict(img, encoder, index):
    debug_print("Starting prediction")
    q = encoder.embed(img)
    beam = beam_search(q, index)
    debug_print(f"Beam size: {len(beam)}")

    cid, pred_lat, pred_lon = final_prediction(beam)
    if cid is None:
        raise ValueError("Beam search returned no final cell")

    debug_print(f"Predicted coordinates: ({pred_lat:.6f}, {pred_lon:.6f})")
    return pred_lat, pred_lon


# ----------------------------
# HISTOGRAM PLOTTING
# ----------------------------
def plot_histogram(distances):
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=50, edgecolor="black", alpha=0.7, color="blue")
    plt.axvline(mean_dist, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_dist:.1f} km")
    plt.axvline(median_dist, color="green", linestyle="--", linewidth=2, label=f"Median: {median_dist:.1f} km")
    plt.title("Full Distribution of Validation Distances")
    plt.xlabel("Distance (km)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    threshold = np.percentile(distances, 50)
    upper_half = distances[distances <= threshold]
    plt.hist(upper_half, bins=25, edgecolor="black", alpha=0.7, color="orange")
    plt.axvline(np.mean(upper_half), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(upper_half):.1f} km")
    plt.axvline(np.median(upper_half), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(upper_half):.1f} km")
    plt.title("Upper 50% of Scores")
    plt.xlabel("Distance (km)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.suptitle("Validation Distance Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# ----------------------------
# EVALUATION
# ----------------------------
def evaluate(encoder, index, progress_callback=None, max_images=None, use_track=True):
    debug_print("Starting evaluation")

    rows = []
    processed_images = 0
    test_images = sorted(TEST_DIR.glob("*"))

    if max_images is not None:
        test_images = test_images[:max_images]

    total_images = len(test_images)
    image_iter = track(test_images) if use_track else test_images

    for i, p in enumerate(image_iter):
        try:
            test_type, idx = parse_test_image_name(p.name)
            gt = get_real_coordinates(test_type)[idx]

            with Image.open(p) as img:
                pred = predict(img.convert("RGB"), encoder, index)

            dist = haversine(gt[0], gt[1], pred[0], pred[1])
            score = geoguessr_score(dist)

            rows.append({
                "img": p.name,
                "dist": dist,
                "score": score,
                "pred_lat": pred[0],
                "pred_lon": pred[1],
                "real_lat": gt[0],
                "real_lon": gt[1],
                "test_type": test_type,
                "image_idx": idx,
            })

            if progress_callback:
                progress = (i + 1) / total_images * 100 if total_images else 100
                progress_callback(progress, f"Processing {p.name} ({i+1}/{total_images})")

            processed_images += 1
        except Exception as e:
            debug_print(f"Error processing {p.name}: {e}")
            print("skip", p, e)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No evaluation rows were generated.")
        return df

    df_mean = df["dist"].mean()
    df_median = df["dist"].median()
    df_score_sum = df["score"].sum()

    print(df.sort_values("dist").head(10))
    print("\nMEAN KM:", df_mean)
    print("MEDIAN:", df_median)
    print("TOTAL IMAGES PROCESSED:", processed_images)
    print("AVERAGE SCORE PER IMAGE:", df_score_sum / processed_images if processed_images > 0 else 0)

    return df


# ----------------------------
# LOAD / BUILD INDEX
# ----------------------------
def load_or_build_index(encoder, index_path=None):
    if index_path is None:
        index_path = get_s2_index_path()

    index_path = Path(index_path)

    if index_path.exists():
        print("Loading cached index...")
        index = load_index(index_path)
        if index is not None:
            return index
        print("Cached index is invalid, rebuilding...")

    print("Building index from scratch...")
    index = build_index(encoder)
    print("Saving index...")
    save_index(index, index_path)
    return index


# ----------------------------
# BENCHMARK
# ----------------------------
def run_benchmark(args):
    print("Running benchmark mode...")
    print(f"Warmup runs: {args.benchmark_warmup}")
    print(f"Timed runs: {args.benchmark_runs}")
    if args.benchmark_max_images is not None:
        print(f"Evaluation image cap: {args.benchmark_max_images}")

    benchmark_rows = []

    t0 = time.perf_counter()
    encoder = SigLIPEncoder()
    benchmark_rows.append(("load_encoder", time.perf_counter() - t0))

    t0 = time.perf_counter()
    index = load_or_build_index(encoder)
    benchmark_rows.append(("load_or_build_index", time.perf_counter() - t0))

    sample_image = get_test_image_path("Game", 0)
    if sample_image is not None and sample_image.exists():
        for _ in range(args.benchmark_warmup):
            with Image.open(sample_image) as img:
                predict(img.convert("RGB"), encoder, index)

        for run_idx in range(args.benchmark_runs):
            t0 = time.perf_counter()
            with Image.open(sample_image) as img:
                predict(img.convert("RGB"), encoder, index)
            benchmark_rows.append((f"predict_game0_run_{run_idx + 1}", time.perf_counter() - t0))

    for _ in range(args.benchmark_warmup):
        evaluate(encoder, index, progress_callback=None, max_images=args.benchmark_max_images, use_track=False)

    for run_idx in range(args.benchmark_runs):
        t0 = time.perf_counter()
        evaluate(encoder, index, progress_callback=None, max_images=args.benchmark_max_images, use_track=False)
        benchmark_rows.append((f"evaluate_run_{run_idx + 1}", time.perf_counter() - t0))

    print("\nBenchmark timings (seconds):")
    for label, seconds in benchmark_rows:
        print(f"- {label:<30} {seconds:.4f}")


# ----------------------------
# ARG PARSING
# ----------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ProtoNet New GeoGuessr Test")
    parser.add_argument("--debug", action="store_true", help="Enable extended debug output")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("--hist", action="store_true", help="Show histogram of validation distances")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Benchmark run count")
    parser.add_argument("--benchmark-warmup", type=int, default=1, help="Benchmark warmup count")
    parser.add_argument("--benchmark-max-images", type=int, default=None, help="Limit evaluation images in benchmark mode")
    return parser.parse_args(argv)


# ----------------------------
# MAIN
# ----------------------------
def main(argv=None):
    args = parse_args(argv)
    configure_debug(args.debug)

    if args.benchmark:
        run_benchmark(args)
        return

    GUI_MODE = args.gui
    if GUI_MODE:
        try:
            from Roy.V2.GUI.gui_main import run_gui
            run_gui()
            return
        except Exception as e:
            print(f"GUI load failed: {e}. Falling back to console mode.")
            GUI_MODE = False

    if not GUI_MODE:
        print("Loading encoder...")
        encoder = SigLIPEncoder()

        print("Building S2 index...")
        index = load_or_build_index(encoder)

        print("Running validation...")
        df = evaluate(encoder, index)

        if args.hist:
            plot_histogram(df["dist"].values)


if __name__ == "__main__":
    main()
