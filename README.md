# Structure-Aware Generative Modeling of Warli Art Motifs from Limited Data Using DCGAN and WGAN-GP

This repository contains the implementation and experimental pipeline for:

**Structure-Aware Generative Modeling of Warli Art Motifs from Limited Data Using DCGAN and WGAN-GP**

## Overview

This study investigates generative learning of structured Warli human motifs under limited-data conditions.

A controlled comparison is performed between:

- Deep Convolutional Generative Adversarial Network (DCGAN)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)

Both architectures are trained using the same Warli "Man" motif dataset, image representation, latent dimensionality, output resolution, training duration, and multi-seed evaluation protocol.

The study focuses on:

- Limited-data generative image modeling
- Distributional alignment
- Structural similarity
- Foreground-aware axial symmetry
- Perceptual diversity
- Training-set proximity
- Multi-seed reproducibility

## Dataset

The experiments use the Warli "Man" motif category.

**Dataset size:** 998 images  
**Image representation:** Single-channel grayscale  
**Image resolution:** 64 × 64 pixels

Images are resized to 64 × 64 pixels, converted to grayscale, transformed to tensors, and normalized approximately to the range [-1, 1].

No offline augmentation, random rotation, affine transformation, or horizontal flipping is used in the final experiments.

The complete 998-image collection is used for training. No separate held-out test set is used.

### Dataset Source

Warli Art Object Image Dataset  
Mendeley Data, 2023.

Only the **"Man"** category is used in the present study.

Please cite the original dataset/publication when using the data.

## Experimental Design

Both DCGAN and WGAN-GP are independently trained using three random seeds:

- 42
- 123
- 2024

This results in six independent training runs:

- 3 DCGAN runs
- 3 WGAN-GP runs

The primary architecture comparison is performed at the common, pre-specified **epoch-100 checkpoint**.

Intermediate checkpoints are retained at:

- Epoch 25
- Epoch 50
- Epoch 75
- Epoch 100

These checkpoints are used to study training progression.

## Common Configuration

| Parameter | Value |
|---|---|
| Training images | 998 |
| Image resolution | 64 × 64 |
| Image channels | 1 |
| Latent dimension | 100 |
| Batch size | 64 |
| Training duration | 100 epochs |
| Seeds | 42, 123, 2024 |
| Primary comparison checkpoint | Epoch 100 |

## DCGAN

### Generator

The generator uses a transposed-convolution architecture with feature-map progression:

100 → 512 → 256 → 128 → 64 → 1

Intermediate layers use:

- Batch normalization
- ReLU activation

The output layer uses:

- Tanh activation

### Discriminator

The discriminator uses convolutional downsampling with channel progression:

1 → 64 → 128 → 256 → 512 → 1

Intermediate layers use:

- Batch normalization
- LeakyReLU activation

The final layer produces raw logits.

### Optimization

- Loss: BCEWithLogitsLoss
- Generator learning rate: 2 × 10⁻⁴
- Discriminator learning rate: 1 × 10⁻⁴
- Adam β₁ = 0.5
- Adam β₂ = 0.999
- Real-label target = 0.9
- One discriminator update per generator update

## WGAN-GP

The WGAN-GP generator uses the same architecture as the DCGAN generator to support a controlled comparison.

### Critic

The critic follows the channel progression:

1 → 64 → 128 → 256 → 512 → 1

The critic uses:

- LeakyReLU activation
- Instance normalization with learnable affine parameters
- No Sigmoid activation

### Optimization

- Wasserstein adversarial objective
- Gradient penalty coefficient λ = 5.0
- Generator learning rate: 2 × 10⁻⁴
- Critic learning rate: 1 × 10⁻⁴
- Adam β₁ = 0
- Adam β₂ = 0.9
- Three critic updates per generator update

## Evaluation Protocol

The primary quantitative comparison is performed at epoch 100 for all three seeds.

The following complementary evaluation measures are used.

### 1. Fréchet Inception Distance (FID)

FID evaluates distributional alignment between the real and generated image distributions.

Lower FID indicates closer alignment in Inception feature space.

Because the original images are grayscale, they are replicated across three channels before Inception feature extraction.

### 2. Random-Reference SSIM

For each evaluated generated image:

1. Five real reference images are randomly selected.
2. SSIM is computed between the generated image and each reference.
3. The five SSIM values are averaged.
4. The procedure is repeated across 100 generated images.

The resulting score provides an image-level structural correspondence measure under a fixed random-reference protocol.

It is not interpreted as reconstruction accuracy.

### 3. Foreground-Aware Axial Symmetry

Axial symmetry is used as a domain-specific structural descriptor for Warli human motifs.

A foreground threshold of:

τ = 0.55

is used.

The metric evaluates left-right foreground correspondence about the vertical image axis.

Higher scores indicate stronger bilateral organization.

This metric is treated as a task-specific descriptor rather than a universal image-quality measure.

### 4. LPIPS-Based Perceptual Diversity

Perceptual diversity is evaluated using LPIPS with a pretrained AlexNet backbone.

For each independently trained epoch-100 model:

- 500 distinct generated-image pairs are sampled.
- LPIPS distance is computed for each pair.
- Mean LPIPS distance is used as the run-level diversity score.

Higher values indicate greater perceptual variability among generated samples.

### 5. SSIM-Based Nearest-Neighbour Analysis

Nearest-neighbour analysis is used as a training-set proximity diagnostic.

For each evaluated generated image, SSIM is calculated against **all 998 training images**.

The training image producing the maximum SSIM is retained as the nearest neighbour.

For each model run, the analysis reports:

- Mean nearest-neighbour SSIM
- Standard deviation
- Minimum
- Maximum
- Corresponding training-image indices

Higher nearest-neighbour SSIM indicates greater proximity to an individual training image.

It should **not** be interpreted by itself as evidence of memorization or superior generative quality.

## Multi-Seed Reporting

Architecture-level results are reported as:

**mean ± sample standard deviation**

across the three independent seeds:

42, 123, and 2024.

Thus, the reported uncertainty in the primary comparison represents **between-seed training variability**.

## Main Results at Epoch 100

Across the three independent seeds:

| Metric | DCGAN | WGAN-GP |
|---|---:|---:|
| FID ↓ | 387.05 ± 14.94 | 375.56 ± 13.29 |
| SSIM ↑ | 0.2796 ± 0.0187 | 0.3173 ± 0.0152 |
| Axial symmetry ↑ | 0.8313 ± 0.0144 | 0.8652 ± 0.0116 |
| LPIPS diversity ↑ | 0.2728 ± 0.0121 | 0.2757 ± 0.0047 |
| Mean NN-SSIM | 0.4871 ± 0.0213 | 0.5705 ± 0.0160 |

WGAN-GP therefore achieves stronger distributional alignment and structural correspondence in the present experimental setting while maintaining comparable perceptual diversity.

The higher nearest-neighbour SSIM of WGAN-GP is interpreted as greater training-set proximity and not necessarily as better generative quality.

## Reproducibility

Random seeds are explicitly applied to:

- Python
- NumPy
- PyTorch
- CUDA

CuDNN deterministic mode is enabled and benchmarking is disabled.

The pipeline stores:

- Model checkpoints
- Training histories
- Per-seed checkpoint metrics
- Generated sample grids
- Symmetry-mask diagnostics
- Nearest-neighbour matches
- Architecture-level summaries
- Exploratory paired statistical comparisons

## Outputs

The pipeline generates:

- Epoch-level training histories
- Generated samples at epochs 25, 50, 75, and 100
- DCGAN training-dynamics plots
- WGAN-GP training-dynamics plots
- Foreground symmetry-mask diagnostics
- SSIM nearest-neighbour comparison figures
- Per-seed metric CSV files
- Multi-seed summary tables
- Exploratory paired comparisons

## Software Requirements

The implementation uses:

- Python
- PyTorch
- torchvision
- NumPy
- pandas
- SciPy
- scikit-image
- torchmetrics
- torch-fidelity
- LPIPS
- Matplotlib

Install the principal dependencies using:

pip install torchmetrics torch-fidelity lpips scikit-image

## Hardware

The experiments are designed for CUDA-enabled GPU execution.

CPU execution is possible but substantially slower.

## Important Interpretation Note

This study is a controlled case study involving one Warli motif category and a limited dataset.

The reported results should therefore not be interpreted as establishing universal superiority of WGAN-GP over DCGAN.

The evaluation measures capture complementary properties:

- FID: distributional alignment
- SSIM: structural correspondence
- Axial symmetry: task-specific bilateral organization
- LPIPS: perceptual diversity
- NN-SSIM: training-set proximity

No single metric is treated as a complete measure of generative quality.
