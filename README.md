# ViT Image Classification

Fine-tune a pre-trained [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224-in21k) on custom image classification tasks using HuggingFace Transformers, and run batch inference on new images.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Preparing Your Data](#preparing-your-data)
- [Pipeline](#pipeline)
  - [Step 1: Fine-tune the Model](#step-1-fine-tune-the-model)
  - [Step 2: Run Predictions on New Images](#step-2-run-predictions-on-new-images)
  - [Step 3: Generate a Confusion Matrix](#step-3-generate-a-confusion-matrix)
- [Running on a Supercomputer (SLURM)](#running-on-a-supercomputer-slurm)
- [Configuration Reference](#configuration-reference)
- [Outputs](#outputs)

## Overview

This project provides a three-step pipeline:

1. **Fine-tuning** — Retrain the classification head (and optionally the backbone) of `google/vit-base-patch16-224-in21k` on your own labeled images.
2. **Prediction** — Run the fine-tuned model on a folder of unlabeled images and save predictions to a CSV file.
3. **Confusion matrix** — Evaluate the fine-tuned model on the labeled training data and produce accuracy, F1 score, and a confusion matrix plot.

The base model (`google/vit-base-patch16-224-in21k`) is pre-trained on ImageNet-21k (14 million images, 21k classes). Fine-tuning replaces the classification head with one matching your number of classes, while keeping the learned image representations from pre-training.

## Project Structure

```
vit/
├── data/
│   ├── images_train/<task>/     # Labeled training images (subfolders per class)
│   └── images_inf/<task>/       # Unlabeled images for inference
├── model/
│   └── google-vit-base-patch16-224-in21k/  # Pre-trained base model (auto-downloaded)
├── output/
│   ├── finetuned_model/<task>/  # Fine-tuned model weights and config
│   ├── prediction/<task>/       # Prediction CSV
│   └── confusion_matrix/<task>/ # Confusion matrix plot
├── script/
│   ├── fine_tuning/
│   │   ├── fine_tuning.py       # Fine-tuning script
│   │   ├── fine_tuning.sh       # SLURM submission script
│   │   └── slurm/               # SLURM log output
│   └── prediction/
│       ├── 00_prediction.py     # Batch inference script
│       ├── 00_prediction.sh     # SLURM submission script
│       ├── 01_confusion_matrix.py  # Evaluation script
│       ├── 01_confusion_matrix.sh  # SLURM submission script
│       └── slurm/               # SLURM log output
├── pyproject.toml               # Python dependencies (managed by uv)
└── .python-version              # Python version (3.11.10)
```

## Prerequisites

- **Python 3.11.10** (exact version required by `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** — Python package and project manager
- **GPU (optional)** — Scripts automatically detect CUDA and fall back to CPU

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd vit
   ```

2. **Create the virtual environment and install dependencies:**

   ```bash
   uv sync
   ```

   This reads `pyproject.toml`, creates a `.venv/` directory with Python 3.11.10, and installs all dependencies. If Python 3.11.10 is not installed on your machine, uv will download it automatically.

3. **Verify the installation:**

   ```bash
   uv run python -c "import transformers; print(transformers.__version__)"
   ```

The base model (`google/vit-base-patch16-224-in21k`) is downloaded automatically on the first run of `fine_tuning.py` and saved to `model/google-vit-base-patch16-224-in21k/`. No manual download is needed.

## Preparing Your Data

### Training images

Organize your labeled images into subfolders inside `data/images_train/<task>/`, where each subfolder name is a class label:

```
data/images_train/skin_color/
├── black/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── mix/
│   ├── image001.jpg
│   └── ...
└── white/
    ├── image001.jpg
    └── ...
```

- Each subfolder name becomes a class label (e.g., `black`, `mix`, `white`).
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
- Every class folder must contain at least one image.
- There is no minimum dataset size, but a few hundred images per class is recommended.

### Inference images

Place the unlabeled images you want to classify into `data/images_inf/<task>/` as a flat folder (no subfolders needed):

```
data/images_inf/skin_color/
├── photo_a.jpg
├── photo_b.png
└── ...
```

## Pipeline

All scripts use relative paths resolved from the repository root, so they work identically on a local machine and on a supercomputer — no path editing is needed.

### Step 1: Fine-tune the Model

**Script:** `script/fine_tuning/fine_tuning.py`

This script:
1. Loads images from `data/images_train/<task>/` using HuggingFace's ImageFolder loader.
2. Splits the data into training and validation sets (85/15 by default).
3. Applies data augmentation (random crop + horizontal flip for training, center crop for validation).
4. Fine-tunes the ViT model for the configured number of epochs.
5. Saves the best model (by validation accuracy) to `output/finetuned_model/<task>/`.

**Run locally:**

```bash
uv run script/fine_tuning/fine_tuning.py
```

**What to expect:** On CPU, a small dataset (~700 images, 4 epochs) takes roughly 40 minutes. On a GPU, the same run completes in a few minutes.

**Before running**, open `script/fine_tuning/fine_tuning.py` and set:
- `TASK_NAME` — your task identifier (e.g., `"skin_color"`). Must match the folder name under `data/images_train/`.

**Optional adjustments** (all at the top of the script):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 16 | Images per GPU per training step |
| `NUM_EPOCHS` | 4 | Full passes over the training set |
| `LEARNING_RATE` | 2e-4 | Step size for the AdamW optimizer |
| `TEST_SIZE` | 0.15 | Fraction of data reserved for validation |
| `SAVE_STEPS` | 100 | Save a checkpoint every N training steps |
| `EVAL_STEPS` | 100 | Run validation every N training steps |
| `LOGGING_STEPS` | 10 | Log training loss every N steps |
| `SAVE_TOTAL_LIMIT` | 2 | Keep only the N most recent checkpoints |

### Step 2: Run Predictions on New Images

**Script:** `script/prediction/00_prediction.py`

This script:
1. Loads the fine-tuned model from `output/finetuned_model/<task>/`.
2. Scans all images in `data/images_inf/<task>/`.
3. Runs inference in batches and computes class probabilities.
4. Saves a CSV file to `output/prediction/<task>/predictions.csv`.

**Run locally:**

```bash
uv run script/prediction/00_prediction.py
```

**Before running**, make sure:
- You have completed Step 1 (the fine-tuned model must exist).
- You have placed images in `data/images_inf/<task>/`.
- `TASK_NAME` in the script matches your task.

**Output CSV format:**

| filename | predicted_label | prob_black | prob_mix | prob_white |
|----------|----------------|------------|----------|------------|
| photo_a.jpg | black | 0.9213 | 0.0512 | 0.0275 |
| photo_b.png | white | 0.0134 | 0.0891 | 0.8975 |

The probability columns are named dynamically based on the class labels in your dataset.

### Step 3: Generate a Confusion Matrix

**Script:** `script/prediction/01_confusion_matrix.py`

This script:
1. Loads the fine-tuned model from `output/finetuned_model/<task>/`.
2. Loads the labeled training images from `data/images_train/<task>/`.
3. Runs inference on all labeled images and compares predictions to true labels.
4. Prints accuracy and weighted F1 score to the terminal.
5. Saves a confusion matrix plot to `output/confusion_matrix/<task>/confusion_matrix.png`.

**Run locally:**

```bash
uv run script/prediction/01_confusion_matrix.py
```

**Before running**, make sure:
- You have completed Step 1 (the fine-tuned model must exist).
- `TASK_NAME` in the script matches your task.

## Running on a Supercomputer (SLURM)

Each Python script has a matching `.sh` file configured for HPC usage. The shell scripts handle module loading, environment activation, and HuggingFace cache configuration.

### First-time setup on the supercomputer

1. **Transfer the repository** to your project directory (e.g., `/project/home/p200804/vit/`).

2. **Create the virtual environment:**

   ```bash
   module load env/release/2024.1
   module load Python/3.11.10-GCCcore-13.3.0
   uv sync
   ```

3. **Create SLURM output directories:**

   ```bash
   mkdir -p script/fine_tuning/slurm
   mkdir -p script/prediction/slurm
   ```

4. **Place your training images** in `data/images_train/<task>/` following the folder structure described above.

### Submitting jobs

All `sbatch` commands must be run from the repository root:

```bash
cd /project/home/p200804/vit/

# Step 1: Fine-tune
sbatch script/fine_tuning/fine_tuning.sh

# Step 2: Predict (after fine-tuning completes)
sbatch script/prediction/00_prediction.sh

# Step 3: Confusion matrix (after fine-tuning completes)
sbatch script/prediction/01_confusion_matrix.sh
```

### Monitoring jobs

```bash
# Check job status
squeue -u $USER

# View live output
tail -f script/fine_tuning/slurm/fine_tuning.out
tail -f script/prediction/slurm/00_prediction.out
tail -f script/prediction/slurm/01_confusion_matrix.out
```

### SLURM resource allocation

| Script | GPUs | CPUs | Time limit | Partition |
|--------|------|------|------------|-----------|
| `fine_tuning.sh` | 4 | 32 | X hours | gpu |
| `00_prediction.sh` | 1 | 32 | X hours | gpu |
| `01_confusion_matrix.sh` | 1 | 32 | X hours | gpu |

Adjust the `--time` parameter in the `.sh` files if your dataset is significantly larger or smaller.

## Configuration Reference

To adapt this project to a new classification task:

1. **Pick a task name** (e.g., `"age"`, `"breed"`, `"defect_type"`).
2. **Create the data folders:**
   ```bash
   mkdir -p data/images_train/<task_name>/<class_1>
   mkdir -p data/images_train/<task_name>/<class_2>
   mkdir -p data/images_inf/<task_name>
   ```
3. **Place your images** in the corresponding folders.
4. **Update `TASK_NAME`** in all three Python scripts:
   - `script/fine_tuning/fine_tuning.py`
   - `script/prediction/00_prediction.py`
   - `script/prediction/01_confusion_matrix.py`
5. **Run the pipeline** (Steps 1-3 above).

No other code changes are needed — the scripts dynamically detect class labels from subfolder names and configure the model accordingly.

## Outputs

After running the full pipeline, your `output/` directory will contain:

```
output/
├── finetuned_model/<task>/          # Fine-tuned model
│   ├── config.json                  # Model architecture and label mapping
│   ├── model.safetensors            # Model weights
│   ├── preprocessor_config.json     # Image processor settings
│   ├── training_args.bin            # Training configuration
│   ├── trainer_state.json           # Training state and history
│   ├── train_results.json           # Training metrics
│   ├── eval_results.json            # Evaluation metrics
│   └── runs/                        # TensorBoard logs
├── prediction/<task>/
│   └── predictions.csv              # Inference results with probabilities
└── confusion_matrix/<task>/
    └── confusion_matrix.png         # Evaluation plot
```

To visualize training metrics with TensorBoard:

```bash
uv run tensorboard --logdir output/finetuned_model/<task>/runs
```
