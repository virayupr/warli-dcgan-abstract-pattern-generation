"""Post-training enhancement analysis used by the revised Warli study.

This script does not retrain either GAN. It consumes the six epoch-100
generated arrays produced by the multi-seed pipeline and reproduces:

1. foreground-threshold sensitivity for symmetry and component descriptors;
2. LPIPS nearest-neighbour retrieval against all real training images; and
3. Inception-v3 cosine nearest-neighbour retrieval.

Expected generated filenames are, for example,
``DCGAN_seed42_generated.npy`` and ``WGAN_GP_seed42_generated.npy``.
Arrays must have shape (N, 64, 64) or (N, 1, 64, 64) and values in [0, 1].
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from torchvision.models import Inception_V3_Weights, inception_v3


SEEDS = (42, 123, 2024)
ARCHITECTURES = ("DCGAN", "WGAN-GP")
THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65)
SMALL_COMPONENT_MAX_PIXELS = 8
EVALUATION_SEED = 2026
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_real_images(root: Path, size: int = 64) -> np.ndarray:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found below {root}")
    images = []
    for path in paths:
        image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(images)


def load_generated(root: Path, architecture: str, seed: int) -> np.ndarray:
    prefix = architecture.replace("-", "_")
    path = root / f"{prefix}_seed{seed}_generated.npy"
    values = np.load(path).astype(np.float32)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError(f"Unexpected generated-array shape in {path}: {values.shape}")
    return np.clip(values, 0.0, 1.0)


def axial_symmetry(image: np.ndarray, threshold: float) -> float:
    mirrored = image[:, ::-1]
    joint = (image >= threshold) | (mirrored >= threshold)
    if not joint.any():
        return 0.0
    discrepancy = np.abs(image - mirrored)[joint].mean()
    return float(np.clip(1.0 - discrepancy, 0.0, 1.0))


def component_descriptors(image: np.ndarray, threshold: float) -> dict[str, float]:
    mask = image >= threshold
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:]
    foreground = int(mask.sum())
    largest_ratio = float(areas.max() / foreground) if foreground and len(areas) else 0.0
    small_count = int((areas <= SMALL_COMPONENT_MAX_PIXELS).sum())
    return {
        "axial_symmetry": axial_symmetry(image, threshold),
        "n_components": float(count),
        "largest_component_ratio": largest_ratio,
        "small_fragment_count": float(small_count),
        "foreground_fraction": float(mask.mean()),
    }


def threshold_analysis(
    real: np.ndarray,
    generated: dict[tuple[str, int], np.ndarray],
    output: Path,
) -> None:
    image_rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        for index, image in enumerate(real):
            image_rows.append({
                "source": "Real", "seed": -1, "image_index": index,
                "threshold": threshold, **component_descriptors(image, threshold),
            })
        for (architecture, seed), images in generated.items():
            for index, image in enumerate(images):
                image_rows.append({
                    "source": architecture, "seed": seed, "image_index": index,
                    "threshold": threshold, **component_descriptors(image, threshold),
                })

    image_frame = pd.DataFrame(image_rows)
    image_frame.to_csv(output / "threshold_sensitivity_image_level.csv", index=False)
    metrics = [
        "axial_symmetry", "n_components", "largest_component_ratio",
        "small_fragment_count", "foreground_fraction",
    ]
    generated_frame = image_frame[image_frame.seed >= 0]
    per_seed = generated_frame.groupby(["source", "seed", "threshold"])[metrics].mean().reset_index()
    per_seed.to_csv(output / "threshold_sensitivity_per_seed.csv", index=False)

    rows = []
    for (architecture, threshold), group in per_seed.groupby(["source", "threshold"]):
        row: dict[str, float | int | str] = {
            "Architecture": architecture,
            "threshold": threshold,
            "N_seeds": group.seed.nunique(),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_between_seed_SD"] = group[metric].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "threshold_sensitivity_architecture_summary.csv", index=False)

    real_summary = image_frame[image_frame.source == "Real"].groupby("threshold")[metrics].mean().reset_index()
    real_summary.to_csv(output / "threshold_sensitivity_real_reference.csv", index=False)
    plot_threshold_metric(summary, real_summary, "axial_symmetry", "Axial symmetry score", output)
    plot_threshold_metric(summary, real_summary, "n_components", "Number of connected components", output)
    plot_threshold_metric(summary, real_summary, "largest_component_ratio", "Largest-component ratio", output)
    plot_threshold_metric(summary, real_summary, "small_fragment_count", "Small-fragment count", output)
    plot_threshold_metric(summary, real_summary, "foreground_fraction", "Foreground fraction", output)


def plot_threshold_metric(
    generated_summary: pd.DataFrame,
    real_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    if metric != "axial_symmetry":
        axis.plot(real_summary.threshold, real_summary[metric], "k--", label="Real reference")
    for architecture, marker in (("DCGAN", "o"), ("WGAN-GP", "s")):
        current = generated_summary[generated_summary.Architecture == architecture]
        axis.errorbar(
            current.threshold,
            current[f"{metric}_mean"],
            yerr=current[f"{metric}_between_seed_SD"],
            marker=marker,
            capsize=3,
            label=architecture,
        )
    axis.set_xlabel("Foreground threshold τ")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    name = metric.title().replace("_", "_")
    fig.savefig(output / f"Threshold_Sensitivity_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def to_lpips_tensor(images: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(images[:, None]).repeat(1, 3, 1, 1)
    return tensor.mul(2.0).sub(1.0)


def lpips_nearest_distances(
    generated: np.ndarray,
    real: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    real_batch: int = 64,
) -> np.ndarray:
    real_tensor = to_lpips_tensor(real)
    distances = []
    with torch.no_grad():
        for image in to_lpips_tensor(generated):
            best = float("inf")
            for start in range(0, len(real_tensor), real_batch):
                batch = real_tensor[start:start + real_batch].to(device)
                query = image.unsqueeze(0).expand(len(batch), -1, -1, -1).to(device)
                current = model(query, batch).flatten()
                best = min(best, float(current.min().cpu()))
            distances.append(best)
    return np.asarray(distances, dtype=np.float32)


def inception_features(
    images: np.ndarray,
    model: torch.nn.Module,
    weights: Inception_V3_Weights,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    output = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size, None]).repeat(1, 3, 1, 1)
            batch = weights.transforms()(batch).to(device)
            features = model(batch)
            features = torch.nn.functional.normalize(features, dim=1)
            output.append(features.cpu().numpy())
    return np.concatenate(output)


def learned_neighbour_analysis(
    real: np.ndarray,
    generated: dict[tuple[str, int], np.ndarray],
    output: Path,
    count: int,
    device: torch.device,
) -> None:
    rng = np.random.default_rng(EVALUATION_SEED)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights)
    inception.fc = torch.nn.Identity()
    inception = inception.to(device).eval()
    real_features = inception_features(real, inception, weights, device)

    rows = []
    for (architecture, seed), images in generated.items():
        indices = rng.choice(len(images), size=min(count, len(images)), replace=False)
        selected = images[indices]
        lpips_values = lpips_nearest_distances(selected, real, lpips_model, device)
        generated_features = inception_features(selected, inception, weights, device)
        cosine_values = (1.0 - generated_features @ real_features.T).min(axis=1)
        rows.append({
            "Architecture": architecture,
            "seed": seed,
            "LPIPS_NN_mean": lpips_values.mean(),
            "LPIPS_NN_minimum": lpips_values.min(),
            "Inception_NN_mean": cosine_values.mean(),
            "Inception_NN_minimum": cosine_values.min(),
        })

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output / "learned_nearest_neighbour_per_seed.csv", index=False)
    summary_rows = []
    for architecture, group in per_seed.groupby("Architecture"):
        summary_rows.append({
            "Architecture": architecture,
            "N_seeds": group.seed.nunique(),
            "LPIPS_NN_mean": group.LPIPS_NN_mean.mean(),
            "LPIPS_NN_between_seed_SD": group.LPIPS_NN_mean.std(ddof=1),
            "Inception_NN_mean": group.Inception_NN_mean.mean(),
            "Inception_NN_between_seed_SD": group.Inception_NN_mean.std(ddof=1),
        })
    pd.DataFrame(summary_rows).to_csv(output / "learned_nearest_neighbour_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-count", type=int, default=500)
    parser.add_argument("--nn-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(EVALUATION_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = load_real_images(args.real_dir)
    generated = {}
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            images = load_generated(args.generated_dir, architecture, seed)
            generated[(architecture, seed)] = images[:args.threshold_count]
    threshold_analysis(real, generated, args.output_dir)
    learned_neighbour_analysis(real, generated, args.output_dir, args.nn_count, device)
    print(f"Analysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
