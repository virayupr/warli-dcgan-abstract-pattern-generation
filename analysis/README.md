# Post-training enhancement analysis

`enhancement_analysis.py` reproduces the analyses added for the revised
manuscript without retraining DCGAN or WGAN-GP. It expects the six epoch-100
generated arrays created by the multi-seed notebook and the 998-image real
dataset.

Example:

```bash
python analysis/enhancement_analysis.py \
  --real-dir /path/to/Dataset/train \
  --generated-dir /path/to/generated_arrays \
  --output-dir outputs/enhancement_full
```

The script evaluates all three seeds (42, 123, and 2024), performs foreground
threshold sensitivity analysis at 0.45--0.65, and computes LPIPS and
Inception-feature nearest-neighbour distances. Foreground is defined by the
bright-pixel rule `I >= threshold`. Inspect the generated mask diagnostics or
the supplied diagnostics in `results/paper_ready/figures` before interpreting
the structural descriptors.

Generated arrays and model checkpoints are intentionally excluded from Git;
they are large, reproducible outputs of the Colab training notebook.
