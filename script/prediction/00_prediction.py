### Batch Prediction with a Fine-tuned ViT Model
###
### Objective:
### Run inference on unlabeled images using a fine-tuned ViT model and save
### the predicted labels and class probabilities to a CSV file.
###
### Methodology:
### 1. Load the fine-tuned model from FINETUNED_MODEL_DIR
### 2. Load all images from INF_DIR (flat folder, no class subfolders needed)
### 3. Preprocess each image (resize, center crop, normalize)
### 4. Run inference in batches and collect predicted labels + probabilities
### 5. Save results to CSV
###
### Input:
### - Unlabeled images at INF_DIR (e.g., data/images_inf/pet_type/)
### - Fine-tuned model at FINETUNED_MODEL_DIR
###
### Output:
### - CSV file at OUTPUT_FILE with columns: filename, predicted_label, prob_<class>...

import matplotlib
matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm import tqdm
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)

# =============================================================================
# Configuration
# =============================================================================

# Task
TASK_NAME = "pet_type"

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
INF_DIR = PROJECT_DIR / "data" / "images_inf" / TASK_NAME
FINETUNED_MODEL_DIR = PROJECT_DIR / "output" / "finetuned_model" / TASK_NAME
OUTPUT_DIR = PROJECT_DIR / "output" / "prediction" / TASK_NAME
OUTPUT_FILE = OUTPUT_DIR / "predictions.csv"

# Inference hyperparameters
BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# =============================================================================
# Utilities
# =============================================================================

def print_header(title: str, char: str = "=") -> None:
    print(char * 80, flush=True)
    print(title, flush=True)
    print(char * 80, flush=True)


def print_subheader(title: str, char: str = "-") -> None:
    print("", flush=True)
    print(title, flush=True)
    print(char * 80, flush=True)


# =============================================================================
# Model + Preprocessing
# =============================================================================

def load_model_and_processor():
    print_header("Loading Fine-tuned Model")
    print(f"  Model dir: {FINETUNED_MODEL_DIR}", flush=True)

    processor = ViTImageProcessor.from_pretrained(FINETUNED_MODEL_DIR)
    model = ViTForImageClassification.from_pretrained(FINETUNED_MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}", flush=True)

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"  Labels: {labels}", flush=True)

    return model, processor, device, labels


def build_inference_transforms(processor):
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    img_size = processor.size["height"]

    return Compose([
        Resize(img_size),
        CenterCrop(img_size),
        ToTensor(),
        normalize,
    ])


# =============================================================================
# Inference
# =============================================================================

def collect_image_paths():
    print_subheader("Collecting images")
    paths = sorted([
        p for p in INF_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    print(f"  Found {len(paths)} images in {INF_DIR}", flush=True)
    return paths


def run_inference(model, transforms, device, labels, image_paths):
    print_header("Running Inference")

    softmax = torch.nn.Softmax(dim=1)
    results = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Predicting"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        batch_tensors = []

        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            batch_tensors.append(transforms(img))

        pixel_values = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = softmax(outputs.logits).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        for j, path in enumerate(batch_paths):
            row = {
                "filename": path.name,
                "predicted_label": labels[preds[j]],
            }
            for k, label in enumerate(labels):
                row[f"prob_{label}"] = round(float(probs[j][k]), 4)
            results.append(row)

    return results


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print_header(f"ViT Prediction: {TASK_NAME}")
    print(f"Start: {datetime.now()}", flush=True)

    # Model
    model, processor, device, labels = load_model_and_processor()
    transforms = build_inference_transforms(processor)

    # Images
    image_paths = collect_image_paths()
    if not image_paths:
        print("  No images found. Exiting.", flush=True)
        return

    # Predict
    results = run_inference(model, transforms, device, labels, image_paths)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(results)
    df.write_csv(OUTPUT_FILE)
    print(f"  Saved {df.height} predictions to {OUTPUT_FILE}", flush=True)

    # Done
    print_header("Complete!")
    print(f"End: {datetime.now()}", flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
