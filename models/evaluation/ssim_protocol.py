# evaluation/ssim_protocol.py
# ============================================================
# SSIM Protocol (paper-ready)
# - Computes SSIM between generated images and real images
# - Default protocol: 100 generated × 5 random real matches
# - Reports mean ± std over all pairs
# - Optionally: "best-of-k" SSIM (for each generated image, take best SSIM
#   among k random real candidates) -> matches your earlier "Best-of-20" idea
#
# Usage (example):
#   python evaluation/ssim_protocol.py \
#     --real_dir data/warli_dataset/man \
#     --gen_dir  results/final_1000 \
#     --n_gen 100 --k_real 5 --best_of 0
#
# Notes:
# - This expects grayscale or RGB images; it converts to grayscale internally.
# - Images are resized to a common size (default 128).
# ============================================================

from __future__ import annotations
import os
import glob
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def load_gray01(path: str, size: int = 128) -> np.ndarray:
    """
    Load image -> grayscale -> resize -> float32 in [0,1].
    """
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize((size, size), resample=Image.Resampling.NEAREST)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def compute_ssim(a01: np.ndarray, b01: np.ndarray) -> float:
    """
    SSIM on grayscale images in [0,1].
    """
    # data_range must match image range
    return float(ssim(a01, b01, data_range=1.0))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def protocol_random_pairs(
    real_paths: List[str],
    gen_paths: List[str],
    n_gen: int = 100,
    k_real: int = 5,
    size: int = 128,
) -> Dict[str, float]:
    """
    Protocol A: 100 generated images × 5 random real pairings (total 500 SSIM values).
    """
    assert len(real_paths) > 0 and len(gen_paths) > 0, "Empty real/gen folder"

    n_gen = min(n_gen, len(gen_paths))
    chosen_gen = random.sample(gen_paths, n_gen)

    vals = []
    for gp in chosen_gen:
        g = load_gray01(gp, size=size)
        chosen_real = random.sample(real_paths, min(k_real, len(real_paths)))
        for rp in chosen_real:
            r = load_gray01(rp, size=size)
            vals.append(compute_ssim(g, r))

    vals = np.array(vals, dtype=np.float32)
    return {
        "protocol": "random_pairs",
        "n_gen": int(n_gen),
        "k_real": int(k_real),
        "n_pairs": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def protocol_best_of_k(
    real_paths: List[str],
    gen_paths: List[str],
    n_gen: int = 100,
    best_of: int = 20,
    size: int = 128,
) -> Dict[str, float]:
    """
    Protocol B: For each generated image, compare with 'best_of' random real images
    and keep the maximum SSIM. This yields 100 SSIM values (one per generated sample).

    This is useful to report "Best-of-20 SSIM".
    """
    assert len(real_paths) > 0 and len(gen_paths) > 0, "Empty real/gen folder"

    n_gen = min(n_gen, len(gen_paths))
    chosen_gen = random.sample(gen_paths, n_gen)

    best_vals = []
    for gp in chosen_gen:
        g = load_gray01(gp, size=size)
        chosen_real = random.sample(real_paths, min(best_of, len(real_paths)))
        svals = []
        for rp in chosen_real:
            r = load_gray01(rp, size=size)
            svals.append(compute_ssim(g, r))
        best_vals.append(max(svals))

    best_vals = np.array(best_vals, dtype=np.float32)
    return {
        "protocol": "best_of_k",
        "n_gen": int(n_gen),
        "best_of": int(best_of),
        "n_scores": int(best_vals.size),
        "mean": float(best_vals.mean()),
        "std": float(best_vals.std(ddof=1)) if best_vals.size > 1 else 0.0,
        "min": float(best_vals.min()),
        "max": float(best_vals.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True, help="Folder with real motif images")
    ap.add_argument("--gen_dir", type=str, required=True, help="Folder with generated images")
    ap.add_argument("--n_gen", type=int, default=100, help="Number of generated images to sample")
    ap.add_argument("--k_real", type=int, default=5, help="Number of random real pairings per generated image")
    ap.add_argument("--best_of", type=int, default=0, help="If >0, run best-of-k protocol (k=best_of)")
    ap.add_argument("--size", type=int, default=128, help="Resize all images to size×size before SSIM")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out_csv", type=str, default="", help="Optional: write summary metrics to a CSV file")
    args = ap.parse_args()

    seed_everything(args.seed)

    real_paths = list_images(args.real_dir)
    gen_paths = list_images(args.gen_dir)

    if len(real_paths) == 0:
        raise FileNotFoundError(f"No images found in real_dir: {args.real_dir}")
    if len(gen_paths) == 0:
        raise FileNotFoundError(f"No images found in gen_dir: {args.gen_dir}")

    print(f"Real images: {len(real_paths)} | Generated images: {len(gen_paths)}")

    summary = protocol_random_pairs(
        real_paths, gen_paths,
        n_gen=args.n_gen,
        k_real=args.k_real,
        size=args.size
    )
    print("\nProtocol A — Random pairs")
    print(f"SSIM: {summary['mean']:.3f} ± {summary['std']:.3f}  "
          f"(n_pairs={summary['n_pairs']}, min={summary['min']:.3f}, max={summary['max']:.3f})")

    best_summary = None
    if args.best_of and args.best_of > 0:
        best_summary = protocol_best_of_k(
            real_paths, gen_paths,
            n_gen=args.n_gen,
            best_of=args.best_of,
            size=args.size
        )
        print("\nProtocol B — Best-of-k")
        print(f"Best-of-{best_summary['best_of']} SSIM: {best_summary['mean']:.3f} ± {best_summary['std']:.3f}  "
              f"(n_scores={best_summary['n_scores']}, min={best_summary['min']:.3f}, max={best_summary['max']:.3f})")

    # Optional CSV output (summary only)
    if args.out_csv:
        rows = [summary]
        if best_summary is not None:
            rows.append(best_summary)
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(args.out_csv, index=False)
        print("\nSaved summary CSV:", args.out_csv)


if __name__ == "__main__":
    main()
