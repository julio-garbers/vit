# Inference Images

Place unlabeled images here that you want the fine-tuned model to classify.

## Expected structure

```
images_inf/skin_color/
├── image1.jpg
├── image2.png
└── ...
```

## Rules

- Images go directly in this folder (no subfolders needed).
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
- The fine-tuned model must exist in `output/finetuned_model/skin_color/` before running predictions (run `fine_tuning.py` first).
- Results are saved to `output/prediction/skin_color/predictions.csv` with the predicted label and class probabilities for each image.
