### Evaluate Fine-tuned ViT Model with Confusion Matrix
###
### Objective:
### Evaluate the fine-tuned model on labeled training data to assess
### classification performance via confusion matrix, accuracy, and F1 score.
###
### Methodology:
### 1. Load the fine-tuned model from FINETUNED_MODEL_DIR
### 2. Load labeled images from DATA_DIR (subfolders per class)
### 3. Preprocess and run inference on all images in batches
### 4. Compute confusion matrix, accuracy, and weighted F1 score
### 5. Save confusion matrix plot and metrics to OUTPUT_DIR
###
### Input:
### - Labeled images at DATA_DIR (e.g., data/images_train/pet_type/{cat,dog,bird}/)
### - Fine-tuned model at FINETUNED_MODEL_DIR
###
### Output:
### - Confusion matrix plot saved to OUTPUT_DIR/confusion_matrix.png
### - Metrics printed to stdout (accuracy, F1)

import matplotlib
matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
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
DATA_DIR = PROJECT_DIR / "data" / "images_train" / TASK_NAME
FINETUNED_MODEL_DIR = PROJECT_DIR / "output" / "finetuned_model" / TASK_NAME
OUTPUT_DIR = PROJECT_DIR / "output" / "confusion_matrix" / TASK_NAME

# Inference hyperparameters
BATCH_SIZE = 8


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
# Model + Data
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

    return model, processor, device


def load_labeled_dataset(processor):
    print_header("Loading Labeled Dataset")
    print(f"  Data dir: {DATA_DIR}", flush=True)

    ds = load_dataset("imagefolder", data_dir=str(DATA_DIR), split="train")
    labels = ds.features["label"].names
    print(f"  Total images: {len(ds)}", flush=True)
    print(f"  Labels: {labels}", flush=True)

    def transform(batch):
        inputs = processor(
            [x.convert("RGB") for x in batch["image"]],
            return_tensors="pt",
        )
        inputs["label"] = batch["label"]
        return inputs

    ds.set_transform(transform)
    return ds, labels


# =============================================================================
# Inference
# =============================================================================

def run_predictions(model, device, ds, labels):
    print_header("Running Predictions")

    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    all_preds = []
    all_true = []

    for batch in tqdm(loader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

        all_preds.extend([labels[p] for p in preds])
        all_true.extend([labels[t] for t in batch["label"].tolist()])

    return all_true, all_preds


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_and_save(all_true, all_preds, labels):
    print_header("Evaluation Results")

    # Metrics
    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="weighted")
    print(f"  Accuracy: {acc:.4f}", flush=True)
    print(f"  F1 (weighted): {f1:.4f}", flush=True)

    # Confusion matrix
    print_subheader("Confusion Matrix")
    cm = confusion_matrix(all_true, all_preds, labels=labels)
    print(cm, flush=True)

    # Plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="OrRd", xticks_rotation=25, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=24)
    ax.tick_params(axis="both", labelsize=15)

    output_file = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to: {output_file}", flush=True)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print_header(f"ViT Confusion Matrix: {TASK_NAME}")
    print(f"Start: {datetime.now()}", flush=True)

    # Model
    model, processor, device = load_model_and_processor()

    # Data
    ds, labels = load_labeled_dataset(processor)

    # Predict
    all_true, all_preds = run_predictions(model, device, ds, labels)

    # Evaluate
    evaluate_and_save(all_true, all_preds, labels)

    # Done
    print_header("Complete!")
    print(f"End: {datetime.now()}", flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
