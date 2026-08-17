import argparse
import copy
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

# Make the project root importable so we can reuse ProtoNet_New directly.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Roy.V2.ProtoNet_New as pn


DEFAULT_LEVELS = [3, 7, 11, 15]
DEFAULT_BEAM_SIZE = 100
DEFAULT_TOP_K = 10
DEFAULT_LEVEL_WEIGHTS = {3: 0.12, 7: 0.20, 11: 0.28, 15: 0.40}


def build_weight_profiles(levels):
    base = {3: 0.12, 7: 0.20, 11: 0.28, 15: 0.40}
    profiles = [
        ("default", {lvl: base.get(lvl, 0.20) for lvl in levels}),
        ("deep_focus", {3: 0.08, 7: 0.15, 11: 0.27, 15: 0.50} if levels == [3, 7, 11, 15] else {lvl: 0.15 for lvl in levels}),
        ("balanced", {3: 0.18, 7: 0.22, 11: 0.3, 15: 0.3} if levels == [3, 7, 11, 15] else {lvl: 1 / len(levels) for lvl in levels}),
        ("shallow_focus", {3: 0.25, 7: 0.3, 11: 0.25, 15: 0.2} if levels == [3, 7, 11, 15] else {lvl: 0.2 for lvl in levels}),
        ("coarse_then_deep", {3: 0.14, 7: 0.18, 11: 0.22, 15: 0.46} if levels == [3, 7, 11, 15] else {lvl: 0.2 for lvl in levels}),
    ]

    normalized = []
    for name, mapping in profiles:
        if not mapping:
            continue
        total = sum(mapping.values())
        if total == 0:
            continue
        normalized.append((name, {k: float(v / total) for k, v in mapping.items()}))

    # Deduplicate by value to keep the search compact.
    seen = set()
    unique = []
    for name, mapping in normalized:
        key = tuple(sorted(mapping.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, mapping))

    return unique


def generate_trials():
    level_options = [
        [3, 7, 11, 15],
        [3, 7, 11],
        [3, 7, 15],
        [3, 11, 15],
        [3, 7, 9, 11, 15],
        [3, 7, 11, 13, 15],
    ]

    beam_sizes = [16, 32, 64, 100, 128]
    top_k_values = [3, 5, 8, 10, 12, 15]

    trials = []
    for levels in level_options:
        weight_profiles = build_weight_profiles(levels)
        for beam_size in beam_sizes:
            for top_k in top_k_values:
                for profile_name, weights in weight_profiles:
                    trials.append({
                        "LEVELS": list(levels),
                        "BEAM_SIZE": beam_size,
                        "TOP_K_REFINEMENT": top_k,
                        "LEVEL_WEIGHTS": {int(k): float(v) for k, v in weights.items()},
                        "profile_name": profile_name,
                    })
    return trials


def set_trial_params(trial):
    pn.LEVELS = list(trial["LEVELS"])
    pn.BEAM_SIZE = int(trial["BEAM_SIZE"])
    pn.TOP_K_REFINEMENT = int(trial["TOP_K_REFINEMENT"])
    pn.LEVEL_WEIGHTS = {int(k): float(v) for k, v in trial["LEVEL_WEIGHTS"].items()}


def evaluate_trial(trial, max_images=None, use_track=False):
    set_trial_params(trial)

    encoder = pn.SigLIPEncoder()
    index = pn.load_or_build_index(encoder)
    df = pn.evaluate(
        encoder,
        index,
        progress_callback=None,
        max_images=max_images,
        use_track=use_track,
    )

    if df.empty:
        return float("inf"), None

    med = float(df["dist"].median())
    return med, df


def run_tuning(max_images=None, limit_trials=None):
    trials = generate_trials()
    if limit_trials is not None:
        trials = trials[:limit_trials]

    best = {
        "median": float("inf"),
        "trial": None,
        "df": None,
    }

    print(f"Running {len(trials)} hyperparameter combinations...")
    print(f"Validation cap: {max_images}")

    for idx, trial in enumerate(trials, start=1):
        try:
            median, df = evaluate_trial(trial, max_images=max_images, use_track=False)
            print(
                f"[{idx}/{len(trials)}] "
                f"LEVELS={trial['LEVELS']} | BEAM={trial['BEAM_SIZE']} | TOPK={trial['TOP_K_REFINEMENT']} | "
                f"WEIGHTS={trial['LEVEL_WEIGHTS']} | median={median:.3f} km"
            )

            if median < best["median"]:
                best = {"median": median, "trial": trial, "df": df}
        except Exception as exc:
            print(f"Trial failed: {trial} | error={exc}")

    return best


def main():
    parser = argparse.ArgumentParser(description="Tune ProtoNet_New hyperparameters by lowest median distance error.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit validation set size for quicker tuning runs.")
    parser.add_argument("--limit-trials", type=int, default=None, help="Only run the first N trials.")
    args = parser.parse_args()

    best = run_tuning(max_images=args.max_images, limit_trials=args.limit_trials)

    print("\nBEST RESULT")
    if best["trial"] is None:
        print("No valid trial completed.")
        return

    print(f"Median error: {best['median']:.3f} km")
    print(f"LEVELS: {best['trial']['LEVELS']}")
    print(f"BEAM_SIZE: {best['trial']['BEAM_SIZE']}")
    print(f"TOP_K_REFINEMENT: {best['trial']['TOP_K_REFINEMENT']}")
    print(f"LEVEL_WEIGHTS: {best['trial']['LEVEL_WEIGHTS']}")
    print(f"Profile: {best['trial'].get('profile_name', 'custom')}")

    if best["df"] is not None:
        print(best["df"].sort_values("dist").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
