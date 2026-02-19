#!/bin/bash -l
#SBATCH --job-name=fine_tuning
#SBATCH --output=script/fine_tuning/slurm/fine_tuning.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --account=p200804
#SBATCH --qos=default

# =============================================================================
# Paths (run sbatch from the repo root)
# =============================================================================

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}/script/fine_tuning"

# =============================================================================
# Environment
# =============================================================================

module --force purge
module load env/release/2024.1
module load CUDA/12.6.0
module load Python/3.11.10-GCCcore-13.3.0
source /project/home/p200812/vit/.venv/bin/activate

# =============================================================================
# Cache (store HuggingFace models and datasets within the project)
# =============================================================================

export HF_HOME="${PROJECT_DIR}/.cache"

# =============================================================================
# Launch
# =============================================================================

uv run "${SCRIPT_DIR}/fine_tuning.py"
