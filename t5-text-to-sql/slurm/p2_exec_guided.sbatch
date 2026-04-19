#!/bin/bash
#SBATCH --job-name=hw4p2_eg_rerank
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/p2_%x_%j.out
#SBATCH --error=slurm/logs/p2_%x_%j.err

# ============================================================================
# HW4 Part 2 — execution-guided beam reranking on the BASE best checkpoint.
# Generates 16 beams per input, executes each on flight_database.db, and picks
# the highest-scored candidate that does not raise an error. Directly attacks
# the 12.2% of dev queries that fail at execution (F1=0 on those).
# No retraining; inference-time only.
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

# Make sure a checkpoint dir exists under the target experiment_name. The base
# weights live at checkpoints/ft_experiments/base/best; we symlink a sibling
# dir named base_egrerank so the existing loader picks up the same weights
# while we write prediction files under a distinct tag.
mkdir -p checkpoints/ft_experiments/base_egrerank
if [ ! -e checkpoints/ft_experiments/base_egrerank/best ]; then
    ln -s ../base/best checkpoints/ft_experiments/base_egrerank/best
fi

echo "=== HW4 Part 2 exec-guided reranking (base checkpoint) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Timestamp: $(date)"
echo ""

python3 execution_guided_inference.py \
    --finetune \
    --experiment_name base_egrerank \
    --eg_name base_egrerank \
    --num_beams 16 \
    --gen_max_length 512 \
    --batch_size 16 \
    --test_batch_size 8 \
    --bf16 \
    --num_workers 4 \
    --split both

echo ""
echo "=== Exec-guided rerank complete: $(date) ==="
