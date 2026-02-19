# Training Images

Place labeled training images here, organized into one subfolder per class.

## Expected structure

```
images_train/skin_color/
├── black/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── mix/
│   ├── image1.jpg
│   └── ...
└── white/
    ├── image1.jpg
    └── ...
```

## Rules

- Each subfolder name becomes a class label (e.g., `black`, `mix`, `white`).
- Every class folder must contain at least one image.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
- A few hundred images per class is recommended for good results.
- The data is automatically split into training and validation sets (85/15 by default, configurable via `TEST_SIZE` in `fine_tuning.py`).
