#!/bin/bash
#SBATCH --job-name=hw4p2_champ_schema_ls_long
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
# Relaunch of schema+label-smoothing champion with explicit 6h walltime
# (the previous --time-less submission hit SLURM's implicit 1h cap).
# Job was still climbing at epoch 16 when killed (F1 0.7593). Use a
# distinct experiment_name so we don't clobber the truncated checkpoint.
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

echo "=== HW4 Part 2 champion relaunch — schema + ls, 6h walltime ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py \
    --finetune \
    --experiment_name champ_schema_ls_long \
    --use_schema_prompt \
    --bf16 \
    --num_workers 4 \
    --label_smoothing 0.1 \
    --grad_clip 1.0 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --scheduler_type linear \
    --num_warmup_epochs 1 \
    --max_n_epochs 30 \
    --patience_epochs 5 \
    --batch_size 64 \
    --test_batch_size 64 \
    --num_beams 4 \
    --gen_max_length 512 \
    --seed 42

echo ""
echo "=== Champion schema+ls (long) run complete: $(date) ==="
