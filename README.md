Learning Structured Visual Patterns under Data Scarcity
A DCGAN-based Study on Warli Motif Generation
This repository contains the official implementation for:
Learning Structured Visual Patterns from Limited Data: A Case Study on Warli Art Motif Generation

Overview
This project investigates structured visual pattern learning under data scarcity using DCGAN-based generative modeling on Warli "man" motifs.
The study focuses on:
Structural coherence under limited data (~450–800 samples)
Symmetry-aware evaluation
Best-of-K SSIM protocol
Diversity in feature space
Controlled architecture benchmarking

Repository Structure
.
├── data/
│   └── warli_dataset/
│       └── man/                # Real motif images
│
├── models/
│   └── dcgan.py                # Generator & Discriminator
│
├── training/
│   └── train_dcgan.py          # Training script
│
├── evaluation/
│   ├── ssim_protocol.py
│   ├── symmetry_score.py
│   ├── diversity_score.py
│   └── fid_evaluation.py
│
├── notebooks/
│   └── pattern.ipynb           # Reproducible notebook
│
├── results/
│   └── (generated outputs)
│
└── README.md

Dataset

Source:
Warli Art Object Image Dataset (Mendeley Data, 2023)

This study uses only the “man” motif subset.
data/warli_dataset/man/*.png

Images are:
Grayscale
Centered
Resized to 128×128

Installation
git clone https://github.com/YOUR_USERNAME/warli-dcgan.git
cd warli-dcgan

pip install -r requirements.txt

Recommended Python: 3.9–3.11
Tested with PyTorch ≥ 2.0

Training
Train DCGAN
python training/train_dcgan.py \
  --data_root data/warli_dataset \
  --image_size 128 \
  --batch_size 64 \
  --epochs 400 \
  --seed 42

  python training/train_dcgan.py \
  --data_root data/warli_dataset \
  --image_size 128 \
  --batch_size 64 \
  --epochs 400 \
Checkpoints and samples are saved to:
results/<run_name>/

Model Architecture
Generator

Transposed convolution stack

BatchNorm + ReLU

Final Tanh

Latent dimension: 100

Discriminator

Convolutional downsampling

BatchNorm + LeakyReLU

Output: Logits (no sigmoid)

Loss:
BCEWithLogitsLoss (recommended configuration)

If sigmoid=False in Discriminator → use BCEWithLogitsLoss
If sigmoid=True → use BCELoss
Input tensor shape must be (B,1,H,W)
If you see:
expected input to have 1 channels, but got 64
you likely permuted tensor incorrectly.

Evaluation Protocol
We implement structured evaluation tailored to geometric motifs:
Best-of-K SSIM Protocol

For each generated image:
Sample K=5 real images
Compute SSIM
Retain maximum
Report mean ± std across 100 generated samples
Run:
python evaluation/ssim_protocol.py \
  --real_dir data/warli_dataset/man \
  --gen_dir results/final_1000 \
  --n_gen 100 \
  --k_real 5 \
  --best_of 20

  Symmetry Score
Axial symmetry measured via horizontal flip consistency.
python evaluation/symmetry_score.py \
  --img_dir results/final_1000

  Diversity Score

Feature-space diversity using MobileNetV2 embeddings:
Mean pairwise cosine distance
Entropy-normalized diversity score
python evaluation/diversity_score.py \
  --img_dir results/final_1000

  FID
  python evaluation/fid_evaluation.py \
  --real_dir data/warli_dataset/man \
  --gen_dir results/final_1000

  Reproducibility

All experiments use:
Fixed random seed (42)
Deterministic CuDNN
Logged hyperparameters in checkpoint metadata
Saved training curves

Figures

The repository supports automatic generation of:
Real vs Generated panel
Best-of-25 SSIM panel
Training dynamics curves
SSIM distribution plots
Symmetry distribution plots

Hardware

Experiments conducted on:
NVIDIA Tesla T4 GPU
CUDA 12.x
PyTorch 2.x
CPU mode also supported (slower training).
