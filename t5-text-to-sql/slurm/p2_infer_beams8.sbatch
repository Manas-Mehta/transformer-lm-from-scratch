#!/bin/bash
#SBATCH --job-name=hw4p2_infer_beams8
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/p2_%x_%j.out
#SBATCH --error=slurm/logs/p2_%x_%j.err

# ============================================================================
# HW4 Part 2 — wider-beam inference on the BASE best checkpoint.
# Writes to experiment_name=base_beams8 (separate SQL/record files) but
# reuses the base best checkpoint via a symlink, so the original
# t5_ft_base_{dev,test}.{sql,pkl} files are preserved.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/transformer-lm-from-scratch/t5-text-to-sql"
ENV_DIR="${SCRATCH}/conda_envs/nlp_hw4"

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/slurm/logs"
cd "${PROJECT_DIR}"

# Create a sibling checkpoint dir pointing at base/best so the loader finds it
# under experiment_name=base_beams8 without copying weights.
mkdir -p checkpoints/ft_experiments/base_beams8
if [ ! -e checkpoints/ft_experiments/base_beams8/best ]; then
    ln -s ../base/best checkpoints/ft_experiments/base_beams8/best
fi

echo "=== HW4 Part 2 inference with beams=8 on base checkpoint ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py \
    --finetune \
    --skip_train \
    --experiment_name base_beams8 \
    --batch_size 16 \
    --test_batch_size 32 \
    --num_beams 8 \
    --bf16 \
    --gen_max_length 512 \
    --seed 42

echo ""
echo "=== Inference complete: $(date) ==="
