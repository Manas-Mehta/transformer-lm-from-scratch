#!/bin/bash
#SBATCH --job-name=hw4p2_champ
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
# HW4 Part 2 — CHAMPION run (for the leaderboard).
# TEMPLATE: before submitting, edit the --learning_rate + --experiment_name
# below to match the winning config from the sweep, and bump num_beams to 8
# for the final test generation.
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

echo "=== HW4 Part 2 champion run ==="
echo "  Job ID: ${SLURM_JOB_ID}, Node: $(hostname), Time: $(date)"

# EDIT ME: --learning_rate and --experiment_name should match the sweep winner.
python3 train_t5.py \
    --finetune \
    --experiment_name champion \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --scheduler_type linear \
    --num_warmup_epochs 1 \
    --max_n_epochs 50 \
    --patience_epochs 8 \
    --batch_size 16 \
    --test_batch_size 16 \
    --num_beams 8 \
    --gen_max_length 512 \
    --seed 42

echo "=== Champion run complete: $(date) ==="
echo "Submission files to download:"
echo "  results/t5_ft_champion_test.sql"
echo "  records/t5_ft_champion_test.pkl"
