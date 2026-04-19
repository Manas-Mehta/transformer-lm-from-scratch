#!/bin/bash
#SBATCH --job-name=hw4p2_longrun_B
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p2_%x_%j.out
#SBATCH --error=slurm/logs/p2_%x_%j.err

# ============================================================================
# Burst — Longrun B on one of two A100s (c24m170-a100-2).
#   Requests 1 of 2 GPUs so Longrun C can run in parallel on the other.
#   lr=7e-4, cosine, 1ep warmup, 60 ep, patience=12, beams=8, ls=0.05,
#   grad_clip=1.0, seed=13. Our full training-side stack.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/nlp_hw4/t5-text-to-sql"
ENV_DIR="${SCRATCH}/conda_envs/nlp_hw4"

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/slurm/logs"
cd "${PROJECT_DIR}"

echo "=== HW4 Part 2 — Burst Longrun B (A100 40GB, 1/2 on dual-GPU node) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py \
    --finetune \
    --experiment_name longrun_B \
    --learning_rate 7e-4 \
    --weight_decay 0.01 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --max_n_epochs 60 \
    --patience_epochs 12 \
    --batch_size 16 \
    --test_batch_size 32 \
    --num_beams 8 \
    --gen_max_length 512 \
    --label_smoothing 0.05 \
    --grad_clip 1.0 \
    --normalize_whitespace \
    --bf16 \
    --num_workers 4 \
    --seed 13

echo ""
echo "=== Burst Longrun B complete: $(date) ==="
