# ============================================================
# diversity_score.py
# Feature-space Diversity Score using MobileNetV2 embeddings
#
# Computes:
#   1. Mean pairwise cosine distance
#   2. Entropy-based diversity score
#
# Usage:
#   python evaluation/diversity_score.py \
#     --img_dir results/final_1000 \
#     --size 128 \
#     --batch_size 32 \
#     --out_csv results/diversity_summary.csv
# ============================================================

from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ============================================================
# Utility
# ============================================================

def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


class ImageDataset(Dataset):
    def __init__(self, image_paths, size=128):
        self.paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    img_paths = list_images(args.img_dir)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.img_dir}")

    print("Number of images:", len(img_paths))

    dataset = ImageDataset(img_paths, size=args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load pretrained MobileNetV2
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    backbone.classifier = torch.nn.Identity()  # remove classification head
    backbone = backbone.to(device)
    backbone.eval()

    embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = backbone(batch)
            feats = F.normalize(feats, dim=1)  # L2 normalize
            embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    # ============================================================
    # 1. Mean Pairwise Cosine Distance
    # ============================================================

    pairwise_dist = cosine_distances(embeddings)
    mean_pairwise_distance = np.mean(pairwise_dist)

    # ============================================================
    # 2. Entropy-Based Diversity Score
    # ============================================================

    # Compute histogram of distances
    hist, _ = np.histogram(pairwise_dist, bins=50, range=(0, 2), density=True)
    hist = hist + 1e-8  # avoid log(0)

    entropy = -np.sum(hist * np.log(hist))
    entropy_norm = entropy / np.log(len(hist))  # normalized entropy

    print("\nDiversity Metrics")
    print("Mean Pairwise Distance:", round(mean_pairwise_distance, 4))
    print("Entropy-based Diversity Score:", round(entropy_norm, 4))

    # Save CSV
    if args.out_csv:
        df = pd.DataFrame([{
            "num_images": len(img_paths),
            "image_size": args.size,
            "mean_pairwise_distance": mean_pairwise_distance,
            "entropy_diversity_score": entropy_norm
        }])
        df.to_csv(args.out_csv, index=False)
        print("Saved CSV:", args.out_csv)


if __name__ == "__main__":
    main()
