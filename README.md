# Structure-Aware Generative Modeling of Warli Art Motifs from Limited Data Using DCGAN and WGAN-GP

This repository contains the implementation, experimental pipeline, and
reproducibility resources for the study:

**Structure-Aware Generative Modeling of Warli Art Motifs from Limited Data
Using DCGAN and WGAN-GP**

## Overview

This study investigates limited-data generative modeling of structured
Warli human motifs using a controlled comparison of:

- Deep Convolutional Generative Adversarial Network (DCGAN)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)

Both architectures are trained using the same Warli "Man" motif dataset,
image representation, latent dimensionality, output resolution, training
duration, checkpoint schedule, and multi-seed evaluation protocol.

The study adopts a multi-perspective, domain-aware evaluation strategy
because no single metric fully characterizes the quality of
geometry-dominant symbolic imagery.

The evaluation considers:

- Distributional alignment
- Image-level structural correspondence
- Foreground-aware bilateral organization
- Foreground topology and fragmentation
- Perceptual diversity
- Training-set proximity
- Qualitative structural characteristics
- Multi-seed reproducibility

---

## Dataset

The experiments use the Warli **"Man"** motif category.

| Property | Configuration |
|---|---|
| Number of images | 998 |
| Motif category | Man |
| Image representation | Single-channel grayscale |
| Image resolution | 64 × 64 pixels |
| Training set | Complete 998-image collection |
| Held-out test set | Not used |
| Offline augmentation | Not used |
| Random rotation | Not used |
| Affine transformation | Not used |
| Horizontal flipping | Not used |

Images are resized to 64 × 64 pixels, converted to grayscale, transformed
to tensors, and normalized approximately to the range [-1, 1].

The source images are photographs of physical Warli motif cards and may
contain variation in card orientation, capture perspective, illumination,
and surrounding background.

No card rectification, background removal, or explicit motif segmentation
is applied in the final training pipeline.

### Dataset Source

**Warli Art Object Image Dataset**  
Mendeley Data, 2023.

Only the **"Man"** category is used in the present study.

Users of the dataset should cite the original dataset/publication.

---

## Experimental Design

Both DCGAN and WGAN-GP are independently trained using three random seeds:

- 42
- 123
- 2024

This results in six independent training runs:

- 3 DCGAN runs
- 3 WGAN-GP runs

The primary architecture-level comparison is performed at the common,
pre-specified **epoch-100 checkpoint**.

Intermediate checkpoints are retained at:

- Epoch 25
- Epoch 50
- Epoch 75
- Epoch 100

These checkpoints are used to examine training progression.

---

## Common Experimental Configuration

| Parameter | DCGAN | WGAN-GP |
|---|---:|---:|
| Training images | 998 | 998 |
| Image resolution | 64 × 64 | 64 × 64 |
| Image channels | 1 | 1 |
| Latent dimension | 100 | 100 |
| Batch size | 64 | 64 |
| Generator learning rate | 2 × 10⁻⁴ | 2 × 10⁻⁴ |
| Discriminator/Critic learning rate | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| Training duration | 100 epochs | 100 epochs |
| Primary checkpoint | Epoch 100 | Epoch 100 |
| Seeds | 42, 123, 2024 | 42, 123, 2024 |

---

## DCGAN

### Generator

The DCGAN generator maps a 100-dimensional latent vector to a 64 × 64
single-channel image using transposed convolutions.

Feature-map progression:

`100 → 512 → 256 → 128 → 64 → 1`

Intermediate layers use:

- Batch normalization
- ReLU activation

The output layer uses:

- Tanh activation

### Discriminator

The discriminator uses convolutional downsampling with channel progression:

`1 → 64 → 128 → 256 → 512 → 1`

Intermediate layers use:

- Batch normalization
- LeakyReLU activation with negative slope 0.2

The final layer produces a raw logit.

### Optimization

- Objective: Binary cross-entropy with logits
- PyTorch loss: `BCEWithLogitsLoss`
- Generator learning rate: 2 × 10⁻⁴
- Discriminator learning rate: 1 × 10⁻⁴
- Adam β₁ = 0.5
- Adam β₂ = 0.999
- Real-label target = 0.9
- Discriminator updates per generator update = 1
- Gradient clipping = 5.0

---

## WGAN-GP

The WGAN-GP generator uses the same latent dimensionality, feature-map
progression, spatial progression, normalization layers, and activation
functions as the DCGAN generator to support a controlled comparison.

### Critic

The critic follows the channel progression:

`1 → 64 → 128 → 256 → 512 → 1`

The critic uses:

- LeakyReLU activation with negative slope 0.2
- Instance normalization with learnable affine parameters
- No Sigmoid activation

The final layer produces an unconstrained scalar critic score.

### Optimization

- Objective: Wasserstein adversarial objective with gradient penalty
- Gradient-penalty coefficient λ = 5.0
- Generator learning rate: 2 × 10⁻⁴
- Critic learning rate: 1 × 10⁻⁴
- Adam β₁ = 0
- Adam β₂ = 0.9
- Critic updates per generator update = 3
- Gradient clipping = 10.0

---

# Evaluation Protocol

The primary quantitative comparison is performed at epoch 100 for all
three independent seeds.

The evaluation combines generic generative metrics with domain-aware
structural descriptors.

The measures are interpreted as complementary rather than interchangeable
indicators of generative quality.

---

## 1. Fréchet Inception Distance (FID)

FID evaluates distributional alignment between the real and generated
image distributions in Inception feature space.

Lower FID indicates closer distributional alignment.

The implementation uses:

`torchmetrics.image.fid.FrechetInceptionDistance`

with:

- `feature = 2048`
- `normalize = True`

Because the source images are grayscale, they are replicated across three
channels before Inception feature extraction.

FID is treated as a comparative distribution-level measure and not as a
complete measure of Warli-specific symbolic correctness.

---

## 2. Random-Reference SSIM

Structural Similarity Index Measure (SSIM) is used as an image-level
structural correspondence descriptor.

For each evaluated generated image:

1. Five real reference images are randomly selected.
2. SSIM is calculated between the generated image and each reference.
3. The resulting similarities are aggregated according to the fixed
   random-reference evaluation protocol.
4. The procedure is applied consistently to both architectures.

SSIM is interpreted as a structural correspondence measure rather than
as reconstruction accuracy or direct symbolic-validity assessment.

---

## 3. Foreground-Aware Axial Symmetry

Foreground-aware axial symmetry is used as a task-specific descriptor of
bilateral organization in the selected Warli human-motif category.

A prespecified primary foreground threshold of:

`τ = 0.55`

is used with the bright-foreground rule `I(x,y) >= τ`, consistently with
the final training and enhancement pipelines. The supplied mask-diagnostic
figures should always be inspected before interpreting structural results.

The metric evaluates left-right foreground correspondence about the
vertical image axis.

Higher values indicate stronger bilateral organization.

Axial symmetry is treated as a domain-specific structural descriptor
rather than as a universal image-quality metric.

---

## 4. Connected-Component Structural Analysis

Connected-component analysis is used to characterize foreground topology
and fragmentation.

The analysis is implemented in:

`notebooks/Warli_Connected_Component_Analysis_NO_RETRAIN.ipynb`

This analysis operates on saved generated samples and therefore does not
require model retraining.

### Foreground Extraction

Images are represented in grayscale over the range [0,1].

A pixel is classified as foreground when:

`I(x,y) >= τ`

with `τ = 0.55` for the primary analysis. This convention matches the
normalization and foreground orientation used by the final Colab pipeline.

Connected foreground regions are identified using **8-neighbour
connectivity**.

### Structural Descriptors

Three complementary descriptors are calculated.

#### Total Connected-Component Count

The number of spatially separated foreground regions in an image.

#### Largest-Component Ratio

The proportion of all foreground pixels belonging to the largest connected
component.

#### Small-Fragment Count

The number of connected components containing **8 pixels or fewer**.

The same thresholding and connected-component procedure is applied to:

- The complete 998-image real dataset
- Epoch-100 DCGAN generated samples
- Epoch-100 WGAN-GP generated samples

The real-image statistics are treated as structural reference values rather
than optimization targets.

Therefore, larger or smaller connected-component values are not
automatically interpreted as better.

Connected-component analysis is a topological diagnostic and does not by
itself establish semantic or symbolic correctness.

---

## 5. LPIPS-Based Perceptual Diversity

Perceptual diversity is evaluated using Learned Perceptual Image Patch
Similarity (LPIPS).

For each independently trained epoch-100 model:

- Generated-image pairs are sampled.
- LPIPS distance is calculated for each pair.
- Mean LPIPS distance is used as the run-level diversity descriptor.

Higher LPIPS distance indicates greater perceptual variability among
generated samples.

LPIPS is used as a diversity descriptor rather than as a direct measure
of Warli-specific structural correctness.

---

## 6. SSIM-Based Nearest-Neighbour Analysis

Nearest-neighbour analysis is used as a training-set proximity diagnostic.

For each evaluated generated image, SSIM is calculated against all
**998 training images**.

The real training image producing the maximum SSIM is retained as the
nearest neighbour.

For each independently trained model, run-level nearest-neighbour
statistics are retained and subsequently aggregated across seeds.

Higher nearest-neighbour SSIM indicates greater proximity to at least one
training image under the adopted SSIM criterion.

It is **not** interpreted by itself as evidence of memorization, novelty,
or superior generative quality.

---

## 7. Learned-Feature Nearest-Neighbour Analysis

The SSIM diagnostic is supplemented using two learned representations:

- AlexNet-based LPIPS nearest-neighbour distance
- Inception-v3 feature cosine nearest-neighbour distance

For each architecture and seed, 100 generated images are retrieved against
all 998 training images. Lower distance indicates greater training-set
proximity in the selected representation. Architecture-level values are
reported as mean ± sample standard deviation across the three seeds.

| Architecture | LPIPS NN distance ↓ | Inception cosine NN distance ↓ |
|---|---:|---:|
| DCGAN | 0.2179 ± 0.0162 | 0.3526 ± 0.0133 |
| WGAN-GP | 0.1972 ± 0.0068 | 0.3238 ± 0.0129 |

Both learned-feature diagnostics agree with NN-SSIM that WGAN-GP outputs
are relatively closer to the training collection. This agreement does not
prove exact replication, memorization, or novelty.

---

## 8. Foreground-Threshold Sensitivity

The structural analysis is repeated at:

`τ = 0.45, 0.50, 0.55, 0.60, 0.65`

using 500 generated images for every architecture and seed and all 998
real images. WGAN-GP retains higher mean axial symmetry at every threshold:

| τ | DCGAN | WGAN-GP |
|---:|---:|---:|
| 0.45 | 0.8289 ± 0.0145 | 0.8622 ± 0.0121 |
| 0.50 | 0.8298 ± 0.0144 | 0.8632 ± 0.0118 |
| 0.55 | 0.8313 ± 0.0144 | 0.8652 ± 0.0116 |
| 0.60 | 0.8338 ± 0.0144 | 0.8686 ± 0.0116 |
| 0.65 | 0.8376 ± 0.0146 | 0.8740 ± 0.0117 |

The symmetry ranking is stable, whereas the architecture closer to the
real connected-component references varies for some descriptors and
thresholds. Topology results should therefore be interpreted together
with the full sensitivity tables.

---

## 9. Qualitative Structural Inspection

Quantitative evaluation is complemented by qualitative inspection of
representative generated motifs.

The inspection considers:

- Presence of head-like circular components
- Formation and closure of triangular body regions
- Continuity of line-based limbs
- Separation or unintended merging of components
- Approximate bilateral organization
- Foreground fragmentation
- Background artifacts
- Variation in pose, orientation, and component placement

The qualitative analysis is descriptive and is not treated as a formal
domain-expert or human-subject evaluation.

---

# Multi-Seed Reporting

Architecture-level results are reported as:

**mean ± sample standard deviation**

across the three independently trained models:

`42, 123, 2024`

The reported standard deviations therefore represent **between-seed
training variability**, rather than variation among individual generated
images within a single run.

Because only three independent seeds are used, inferential statistical
comparisons should be regarded as exploratory.

---

# Main Results at Epoch 100

## Generic and Structure-Oriented Metrics

| Metric | DCGAN | WGAN-GP |
|---|---:|---:|
| FID ↓ | 387.05 ± 14.94 | **375.56 ± 13.29** |
| Random-reference SSIM ↑ | 0.2796 ± 0.0187 | **0.3173 ± 0.0152** |
| Axial symmetry ↑ | 0.8313 ± 0.0144 | **0.8652 ± 0.0116** |
| LPIPS diversity | 0.2728 ± 0.0121 | 0.2757 ± 0.0047 |
| Mean NN-SSIM | 0.4871 ± 0.0213 | 0.5705 ± 0.0160 |

At the common epoch-100 checkpoint, WGAN-GP achieves lower mean FID and
higher random-reference SSIM and axial symmetry than DCGAN.

LPIPS-based perceptual diversity remains comparable between the two
architectures.

WGAN-GP also exhibits higher nearest-neighbour SSIM, indicating greater
training-set proximity under the adopted similarity measure rather than
necessarily superior generative quality.

LPIPS and Inception-feature nearest-neighbour retrieval produce the same
relative proximity direction. WGAN-GP's axial-symmetry advantage also
remains stable over the complete tested threshold range from 0.45 to 0.65.

---

## Connected-Component Findings

Connected-component analysis reveals complementary structural behaviour
rather than uniform dominance by either architecture.

DCGAN more closely matches the real-image reference in:

- Total connected-component count
- Largest-component ratio

WGAN-GP more closely reproduces the real-image reference in:

- Small-fragment count

For small-fragment count:

- Real images: **1.441**
- DCGAN: **1.982 ± 0.497**
- WGAN-GP: **1.391 ± 0.333**

These findings demonstrate why foreground topology should be considered
alongside FID, SSIM, symmetry, LPIPS diversity, nearest-neighbour
similarity, and qualitative inspection.

---

# Interpretation of the Results

The results do not establish universal superiority of WGAN-GP over DCGAN.

Instead, they reveal complementary model behaviour.

WGAN-GP performs more favourably under:

- FID
- Random-reference SSIM
- Foreground-aware axial symmetry
- Small-fragment agreement with the real-image reference

DCGAN more closely matches the real-image reference under:

- Total connected-component count
- Largest-component ratio

LPIPS diversity remains similar between the architectures.

Nearest-neighbour analysis indicates greater training-set proximity for
WGAN-GP.

The results therefore support a **multi-perspective, domain-aware
evaluation strategy** rather than reliance on a single generative-quality
metric.

---

# Reproducibility

Random seeds are explicitly applied to:

- Python
- NumPy
- PyTorch

The final experimental pipeline stores run-specific:

- Model checkpoints
- Training histories
- Generated samples
- Per-seed quantitative metrics
- Training-progression outputs
- Symmetry diagnostics
- Connected-component statistics
- Nearest-neighbour matches
- Architecture-level summaries

Six independent training runs are performed in total.

The `notebooks` directory contains the Colab-ready multi-seed training
pipeline and the no-retraining connected-component analysis. Paper-ready
CSV summaries and figures are stored under `results/paper_ready`.

---

# Repository Outputs

The experimental pipeline produces:

- Epoch-level training histories
- Generator checkpoints
- Generated samples at epochs 25, 50, 75, and 100
- DCGAN training-dynamics plots
- WGAN-GP training-dynamics plots
- Representative training-progression grids
- Foreground symmetry diagnostics
- Connected-component analysis outputs
- Nearest-neighbour comparison figures
- Per-seed metric CSV files
- Multi-seed summary tables
- Architecture-level quantitative summaries

---

# Software Requirements

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

Install the principal evaluation dependencies using:

```bash
pip install torchmetrics torch-fidelity lpips scikit-image
```

---

# Important Interpretation Notes

- The work presents a **multi-perspective evaluation protocol**, not a
  universally validated metric framework.
- The complete 998-image collection is used for training and as the real
  reference; no genuinely held-out test set is claimed.
- Nearest-neighbour results diagnose training-set proximity and cannot by
  themselves establish memorization or novelty.
- Axial symmetry is relevant to the selected human-motif category but is
  not a universal measure of cultural or symbolic validity.
- Connected-component statistics measure foreground topology and do not
  verify head–torso–limb semantics.
