# Training

The canonical training implementation is the Colab-ready notebook:

`notebooks/warli_multi_seed_pipeline_FINAL_COLAB.ipynb`

It performs all six independent runs: DCGAN and WGAN-GP with seeds 42,
123, and 2024. The common primary checkpoint is epoch 100. Use the notebook
because it contains the complete model definitions, deterministic setup,
checkpoint handling, evaluation calls, and export logic used for the paper.

The earlier empty `train_dcgan.py` placeholder has been removed to avoid
suggesting that it is an alternative reproduction path.
