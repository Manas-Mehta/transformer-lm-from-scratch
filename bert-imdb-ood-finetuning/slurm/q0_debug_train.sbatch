#!/bin/bash
#SBATCH --job-name=hw4p1_q0_debug
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/q0_%x_%j.out
#SBATCH --error=slurm/logs/q0_%x_%j.err

# ============================================================================
# NYU NLP HW4 Part 1 — Q1 debug sanity check (required by the PDF).
# Trains on a 4k subset for 3 epochs and evaluates on a 1k subset.
# PDF estimate: ~7 min train + 1 min eval. Target: >= 88% accuracy on debug.
# If this fails, do NOT proceed to the full Q1 job.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/transformer-lm-from-scratch/bert-imdb-ood-finetuning"
ENV_DIR="${SCRATCH}/conda_envs/nlp_hw4"

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/slurm/logs"
cd "${PROJECT_DIR}"

echo "=== HW4 Part 1 Q0 (debug): --train --eval --debug_train ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python3 main.py --train --eval --debug_train

echo ""
echo "=== Q0 debug complete: $(date) ==="
echo "Expected: accuracy > 88% on the 1k debug eval subset."
