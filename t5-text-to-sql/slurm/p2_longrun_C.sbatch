#!/bin/bash
#SBATCH --job-name=hw4p2_longrun_C
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/p2_%x_%j.out
#SBATCH --error=slurm/logs/p2_%x_%j.err

# ============================================================================
# Longrun C — "full stack + schema-in-prompt" long-schedule run.
#   Same recipe as B, plus --use_schema_prompt.
#   Prepends compact DB schema to each encoder input (src cap grows 128→1024).
#   Targets the ~5% "no such column" execution errors on dev.
#   lr=7e-4, cosine, 1ep warmup, 60ep, patience=12
#   wd=0.01, label_smoothing=0.05, grad_clip=1.0, beams=8 eval, seed=21
#   Plan: after this finishes, apply execution-guided reranking on the
#   resulting best checkpoint.
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

echo "=== HW4 Part 2 — Longrun C (full stack + schema-in-prompt) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py \
    --finetune \
    --experiment_name longrun_C \
    --learning_rate 7e-4 \
    --weight_decay 0.01 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --max_n_epochs 60 \
    --patience_epochs 12 \
    --batch_size 16 \
    --test_batch_size 16 \
    --num_beams 8 \
    --gen_max_length 512 \
    --label_smoothing 0.05 \
    --grad_clip 1.0 \
    --use_schema_prompt \
    --normalize_whitespace \
    --bf16 \
    --num_workers 4 \
    --seed 21

echo ""
echo "=== Longrun C complete: $(date) ==="
