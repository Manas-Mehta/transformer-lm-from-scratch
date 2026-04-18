#!/bin/bash
#SBATCH --job-name=hw4p2_lr1e3
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
# HW4 Part 2 — LR sweep variant: lr 1e-3 (more aggressive than base).
# Run in parallel with p2_sweep_lr3e-4.sbatch after base confirms plumbing.
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

echo "=== HW4 Part 2 sweep: lr=1e-3 ==="
echo "  Job ID: ${SLURM_JOB_ID}, Node: $(hostname), Time: $(date)"

python3 train_t5.py \
    --finetune \
    --experiment_name lr1e-3 \
    --learning_rate 1e-3 \
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

echo "=== Sweep lr=1e-3 complete: $(date) ==="
