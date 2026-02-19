#!/bin/bash -l
#SBATCH --job-name=00_prediction
#SBATCH --output=script/prediction/slurm/00_prediction.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --account=p200804
#SBATCH --qos=default

# =============================================================================
# Paths (relative to repo root — run sbatch from there)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# Environment
# =============================================================================

module --force purge
module load env/release/2024.1
module load CUDA/12.6.0
module load Python/3.11.10-GCCcore-13.3.0
source "${PROJECT_DIR}/.venv/bin/activate"

# =============================================================================
# Cache (store HuggingFace models and datasets within the project)
# =============================================================================

export HF_HOME="${PROJECT_DIR}/.cache"

# =============================================================================
# Launch
# =============================================================================

uv run "${SCRIPT_DIR}/00_prediction.py"
