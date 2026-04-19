#!/bin/bash
#SBATCH --job-name=hw4p2_resume_B
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/p2_%x_%j.out
#SBATCH --error=slurm/logs/p2_%x_%j.err

# ============================================================================
# Burst — Resume B. Warm-start from longrun_B/best (F1 0.8250 @ epoch 15).
# Small LR, no warmup, linear decay. Runs on A100 40GB (Burst side).
# num_beams 4 during train eval. Final test preds regen separately at beams=8.
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

echo "=== HW4 Part 2 — Burst Resume B (warm-start from 0.8250, A100 40GB) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py \
    --finetune \
    --resume_best \
    --experiment_name longrun_B \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --scheduler_type linear \
    --num_warmup_epochs 0 \
    --max_n_epochs 12 \
    --patience_epochs 6 \
    --batch_size 16 \
    --test_batch_size 32 \
    --num_beams 4 \
    --gen_max_length 512 \
    --label_smoothing 0.05 \
    --grad_clip 1.0 \
    --normalize_whitespace \
    --bf16 \
    --num_workers 4 \
    --seed 13

echo ""
echo "=== Burst Resume B complete: $(date) ==="
