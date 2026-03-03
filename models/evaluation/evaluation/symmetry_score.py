# ============================================================
# symmetry_score.py
# Axial Symmetry Score (Foreground-aware, noise tolerant)
#
# Computes vertical symmetry score:
#
#   S = 1 − mean(|I − mirror(I)|)   (masked foreground version)
#
# Inputs:
#   --img_dir  : folder with images
#   --size     : resize size (default 128)
#   --threshold: foreground threshold (default 0.5)
#   --top_k    : optional, export top-k most symmetric images
#   --out_csv  : optional CSV summary
#
# Usage (example):
#   python evaluation/symmetry_score.py \
#     --img_dir results/final_1000 \
#     --size 128 \
#     --threshold 0.5 \
#     --top_k 25 \
#     --out_csv results/symmetry_summary.csv
# ============================================================

from __future__ import annotations
import os
import glob
import argparse
import numpy as np
from PIL import Image
import pandas as pd


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def load_gray01(path, size=128):
    """
    Load image -> grayscale -> resize -> float32 [0,1]
    """
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize((size, size), resample=Image.Resampling.NEAREST)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def compute_symmetry_score(I, threshold=0.5):
    """
    Foreground-aware axial symmetry score.

    Steps:
    1. Create binary mask M (foreground)
    2. Mirror image horizontally
    3. Create joint mask J = M OR M_mirror
    4. Compute masked mean absolute difference
    5. Return 1 - discrepancy
    """

    # Binary foreground mask
    M = (I > threshold).astype(np.float32)

    # Horizontal mirror
    I_m = np.fliplr(I)
    M_m = np.fliplr(M)

    # Joint mask
    J = np.logical_or(M, M_m).astype(np.float32)

    # Avoid division by zero
    denom = np.sum(J)
    if denom == 0:
        return 0.0

    diff = np.abs(I - I_m)
    masked_diff = diff * J

    discrepancy = np.sum(masked_diff) / denom

    score = 1.0 - discrepancy
    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Folder with images")
    parser.add_argument("--size", type=int, default=128,
                        help="Resize images to size×size")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Foreground threshold (0–1)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Export top-k most symmetric images")
    parser.add_argument("--out_csv", type=str, default="",
                        help="Optional CSV output file")
    args = parser.parse_args()

    img_paths = list_images(args.img_dir)

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.img_dir}")

    print(f"Found {len(img_paths)} images")

    results = []

    for path in img_paths:
        I = load_gray01(path, size=args.size)
        score = compute_symmetry_score(I, threshold=args.threshold)
        results.append({
            "image": os.path.basename(path),
            "symmetry_score": score
        })

    df = pd.DataFrame(results)
    mean_score = df["symmetry_score"].mean()
    std_score = df["symmetry_score"].std(ddof=1)

    print("\nAxial Symmetry Score")
    print(f"Mean ± Std: {mean_score:.3f} ± {std_score:.3f}")
    print(f"Min: {df['symmetry_score'].min():.3f}")
    print(f"Max: {df['symmetry_score'].max():.3f}")

    # Save CSV if requested
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print("\nSaved CSV:", args.out_csv)

    # Export Top-K most symmetric images
    if args.top_k and args.top_k > 0:
        top_df = df.sort_values("symmetry_score", ascending=False).head(args.top_k)

        out_dir = os.path.join(args.img_dir, f"top_{args.top_k}_symmetric")
        os.makedirs(out_dir, exist_ok=True)

        for _, row in top_df.iterrows():
            src = os.path.join(args.img_dir, row["image"])
            dst = os.path.join(out_dir, row["image"])
            Image.open(src).save(dst)

        print(f"\nExported top {args.top_k} symmetric images to:")
        print(out_dir)


if __name__ == "__main__":
    main()
