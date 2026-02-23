# Inference Images

Place unlabeled images here that you want the fine-tuned model to classify.

## Expected structure

```
images_inf/pet_type/
├── image1.jpg
├── image2.png
└── ...
```

## Rules

- Images go directly in this folder (no subfolders needed).
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
- The fine-tuned model must exist in `output/finetuned_model/pet_type/` before running predictions (run `fine_tuning.py` first).
