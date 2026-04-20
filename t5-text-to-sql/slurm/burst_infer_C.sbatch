#!/bin/bash
#SBATCH --job-name=hw4p2_infer_C
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/p2_%x_%j.out
#SBATCH --error=slurm/logs/p2_%x_%j.err

# ============================================================================
# Burst — Inference-only on longrun_C best ckpt (schema-in-prompt model).
# Loads current best/ at submit time, runs dev + test decode with beams=8.
# Must include --use_schema_prompt (C was trained with schema in encoder input).
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

echo "=== HW4 Part 2 — Burst Inference C (schema-in-prompt) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py --finetune --skip_train \
    --experiment_name longrun_C \
    --test_batch_size 16 \
    --num_beams 8 \
    --gen_max_length 512 \
    --use_schema_prompt \
    --normalize_whitespace \
    --bf16 \
    --num_workers 4 \
    --seed 21

echo ""
echo "=== Burst Inference C complete: $(date) ==="
