### Fine-tuning a Vision Transformer (ViT) for Image Classification
###
### Objective:
### Fine-tune a pre-trained google/vit-base-patch16-224-in21k model on a custom
### image classification task using HuggingFace Transformers.
###
### Methodology:
### 1. Load images from subdirectory-labeled folders via HuggingFace datasets ImageFolder
### 2. Split into train and validation sets (configurable via TEST_SIZE;
###    85/15 is a good default for small datasets to maximize training data)
### 3. Apply image transforms: random crop + flip (train), center crop (val)
### 4. Fine-tune with HuggingFace Trainer for N epochs with fp16
### 5. Evaluate and save the best model based on validation accuracy
###
### Input:
### - Image folder at DATA_DIR, organized as subfolders per class label
###   (e.g., data/images_train/pet_type/cat/, data/images_train/pet_type/dog/)
###
### Output:
### - Fine-tuned model saved to MODEL_OUTPUT_DIR
### - Training/eval metrics logged to MODEL_OUTPUT_DIR

from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

# =============================================================================
# Configuration
# =============================================================================

# Task
TASK_NAME = "pet_type"  # enter your task name here (e.g., "pet_type")

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "images_train" / TASK_NAME
MODEL_OUTPUT_DIR = PROJECT_DIR / "output" / "finetuned_model" / TASK_NAME

# Pre-trained base model
BASE_MODEL_HUB_ID = "google/vit-base-patch16-224-in21k"
BASE_MODEL_DIR = PROJECT_DIR / "model" / "google-vit-base-patch16-224-in21k"

# Training hyperparameters
BATCH_SIZE = 16  # number of images per GPU per training step
NUM_EPOCHS = 4  # full passes over the training set
LEARNING_RATE = 2e-4  # step size for the optimizer (AdamW)
SAVE_STEPS = 100  # save a checkpoint every N training steps
EVAL_STEPS = 100  # run validation every N training steps
LOGGING_STEPS = 10  # log training loss every N steps
SAVE_TOTAL_LIMIT = 2  # keep only the N most recent checkpoints on disk
TEST_SIZE = 0.15  # fraction of data reserved for validation


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


def ensure_base_model():
    if BASE_MODEL_DIR.exists() and any(BASE_MODEL_DIR.iterdir()):
        return
    print(f"  Base model not found at {BASE_MODEL_DIR}", flush=True)
    print(f"  Downloading from {BASE_MODEL_HUB_ID}...", flush=True)
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL_HUB_ID)
    model.save_pretrained(BASE_MODEL_DIR)
    processor = ViTImageProcessor.from_pretrained(BASE_MODEL_HUB_ID)
    processor.save_pretrained(BASE_MODEL_DIR)
    print(f"  Saved to {BASE_MODEL_DIR}", flush=True)


# =============================================================================
# Data Preparation
# =============================================================================


def load_and_split_dataset():
    print_header("Data Preparation")

    # Load images from subdirectory-labeled folder
    print_subheader("Loading images from ImageFolder")
    ds = load_dataset("imagefolder", data_dir=str(DATA_DIR), split="train")
    print(f"  Total images: {len(ds)}", flush=True)

    # Train/validation split
    print_subheader(f"Splitting data (test_size={TEST_SIZE})")
    data = ds.train_test_split(test_size=TEST_SIZE)
    print(f"  Train: {len(data['train'])} images", flush=True)
    print(f"  Val:   {len(data['test'])} images", flush=True)

    # Labels
    labels = data["train"].features["label"].names
    print(f"  Labels: {labels}", flush=True)

    return data, labels


# =============================================================================
# Image Transforms
# =============================================================================


def build_transforms(feature_extractor):
    print_subheader("Building image transforms")

    normalize = Normalize(
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std,
    )
    img_size = feature_extractor.size["height"]
    print(f"  Image size: {img_size}x{img_size}", flush=True)

    train_transforms = Compose(
        [
            RandomResizedCrop(img_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ]
    )

    return train_transforms, val_transforms


def apply_transforms(data, train_transforms, val_transforms):
    def preprocess_train(batch):
        batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in batch["image"]
        ]
        return batch

    def preprocess_val(batch):
        batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in batch["image"]
        ]
        return batch

    train_ds = data["train"]
    val_ds = data["test"]

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    return train_ds, val_ds


# =============================================================================
# Model Setup
# =============================================================================


def load_model(labels):
    print_header("Model Setup")

    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    print(f"  Pre-trained model: {BASE_MODEL_DIR}", flush=True)
    print(f"  Num labels: {len(labels)}", flush=True)
    print(f"  Label mapping: {label2id}", flush=True)

    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    return model


# =============================================================================
# Training
# =============================================================================


def train_model(model, feature_extractor, train_ds, val_ds):
    print_header("Training")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch]),
        }

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        eval_strategy="steps",
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataloader_num_workers=8,  # parallel data loading (adjust up to CPUs available)
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    print(f"  Output dir: {MODEL_OUTPUT_DIR}", flush=True)
    print(f"  Batch size: {BATCH_SIZE}", flush=True)
    print(f"  Epochs: {NUM_EPOCHS}", flush=True)
    print(f"  Learning rate: {LEARNING_RATE}", flush=True)
    print("  FP16: True", flush=True)

    train_results = trainer.train()

    # Save model and metrics
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluate
    print_subheader("Evaluation")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save final model
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    print(f"  Model saved to: {MODEL_OUTPUT_DIR}", flush=True)

    return metrics


# =============================================================================
# Main Execution
# =============================================================================


def main():
    print_header(f"ViT Fine-tuning: {TASK_NAME}")
    print(f"Start: {datetime.now()}", flush=True)

    # Base model
    ensure_base_model()

    # Data
    data, labels = load_and_split_dataset()

    # Transforms
    feature_extractor = ViTImageProcessor.from_pretrained(BASE_MODEL_DIR)
    train_transforms, val_transforms = build_transforms(feature_extractor)
    train_ds, val_ds = apply_transforms(data, train_transforms, val_transforms)

    # Model
    model = load_model(labels)

    # Training
    metrics = train_model(model, feature_extractor, train_ds, val_ds)

    # Done
    print_header("Complete!")
    print(f"End: {datetime.now()}", flush=True)
    print(f"Final eval accuracy: {metrics.get('eval_accuracy', 'N/A')}", flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
