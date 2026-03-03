# ============================================================
# fid_evaluation.py
# Fréchet Inception Distance (FID) evaluation
#
# Uses torchmetrics implementation (recommended)
#
# Usage:
#   python evaluation/fid_evaluation.py \
#     --real_dir data/warli_dataset/man \
#     --gen_dir results/final_1000 \
#     --batch_size 32 \
#     --size 128 \
#     --out_csv results/fid_summary.csv
# ============================================================

from __future__ import annotations
import os
import glob
import argparse
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from torchmetrics.image.fid import FrechetInceptionDistance


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ============================================================
# Utility Functions
# ============================================================

def list_images(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, size=128):
        self.paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # returns (C,H,W) in [0,1]
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
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory containing real images")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    real_paths = list_images(args.real_dir)
    gen_paths = list_images(args.gen_dir)

    if len(real_paths) == 0:
        raise FileNotFoundError(f"No images found in real_dir: {args.real_dir}")
    if len(gen_paths) == 0:
        raise FileNotFoundError(f"No images found in gen_dir: {args.gen_dir}")

    print(f"Real images: {len(real_paths)}")
    print(f"Generated images: {len(gen_paths)}")

    real_dataset = ImageFolderDataset(real_paths, size=args.size)
    gen_dataset = ImageFolderDataset(gen_paths, size=args.size)

    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False)

    # Torchmetrics FID
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Update with real images
    for batch in real_loader:
        batch = batch.to(device)
        fid.update(batch, real=True)

    # Update with generated images
    for batch in gen_loader:
        batch = batch.to(device)
        fid.update(batch, real=False)

    fid_value = fid.compute().item()

    print("\nFID Score:", round(fid_value, 3))

    # Optional CSV export
    if args.out_csv:
        df = pd.DataFrame([{
            "num_real": len(real_paths),
            "num_generated": len(gen_paths),
            "image_size": args.size,
            "fid": fid_value
        }])
        df.to_csv(args.out_csv, index=False)
        print("Saved CSV:", args.out_csv)


if __name__ == "__main__":
    main()
