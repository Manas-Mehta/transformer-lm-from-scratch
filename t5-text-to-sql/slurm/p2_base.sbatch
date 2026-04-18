#!/bin/bash
#SBATCH --job-name=hw4p2_base
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/p2_%x_%j.out
#SBATCH --error=slurm/logs/p2_%x_%j.err

# ============================================================================
# NYU NLP HW4 Part 2 — BASE T5 fine-tuning run.
# Config: lr 5e-4, 30 epochs max, patience 5, bs 16, beam 4.
# Goal: clear the 65 Record F1 threshold on dev => full 25/25 performance.
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

echo "=== HW4 Part 2 base run ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py \
    --finetune \
    --experiment_name base \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --scheduler_type linear \
    --num_warmup_epochs 1 \
    --max_n_epochs 30 \
    --patience_epochs 5 \
    --batch_size 16 \
    --test_batch_size 32 \
    --num_beams 4 \
    --gen_max_length 512 \
    --seed 42

echo ""
echo "=== Base run complete: $(date) ==="
