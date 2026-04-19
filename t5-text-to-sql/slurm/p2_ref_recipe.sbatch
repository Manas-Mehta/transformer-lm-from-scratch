#!/bin/bash
#SBATCH --job-name=hw4p2_ref_recipe
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
# Reference-style recipe: longer schedule + whitespace normalization.
#
# Key differences from our previous best (base, dev F1 0.7786):
#   --learning_rate     1e-3   (was 5e-4)
#   --weight_decay      0.0    (was 0.01)
#   --scheduler_type    cosine (was linear)
#   --num_warmup_epochs 0      (was 1)
#   --max_n_epochs      80     (was 30)
#   --patience_epochs   15     (was 5)
#   --num_beams         1      (was 4; greedy)
#   --normalize_whitespace on  (collapses whitespace in NL+SQL to T5's canonical
#                               form — the biggest pipeline change)
# All other knobs kept at our defaults. No label smoothing, no schema prompt.
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

echo "=== HW4 Part 2 — reference recipe (normalize_ws + cosine lr1e-3) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Timestamp: $(date)"
echo ""

python3 train_t5.py \
    --finetune \
    --experiment_name ref_recipe \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --scheduler_type cosine \
    --num_warmup_epochs 0 \
    --max_n_epochs 80 \
    --patience_epochs 15 \
    --batch_size 16 \
    --test_batch_size 16 \
    --num_beams 1 \
    --gen_max_length 512 \
    --normalize_whitespace \
    --bf16 \
    --num_workers 4 \
    --seed 42

echo ""
echo "=== Reference recipe run complete: $(date) ==="
