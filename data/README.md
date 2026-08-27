# Dataset placement

The image data are not redistributed in this repository. Download the
[Warli Art Object Image Dataset](https://doi.org/10.17632/vv6dbrwwnn.1)
(CC BY 4.0) and use the 998 images from its `Man` category.

For local scripts, a suitable layout is:

```text
data/
└── warli_dataset/
    └── man/
        ├── image_0001.png
        └── ...
```

The Colab notebooks expose `DATA_DIR` near the beginning of the configuration
cell. Set it to the corresponding Google Drive folder before running the
pipeline. The study uses all 998 selected images as its training collection
and real reference; it does not claim a held-out test set.
