#!/bin/bash
#SBATCH --job-name=sft_math
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j_sft.out
#SBATCH --error=logs/%j_sft.err
#SBATCH --requeue

# Usage: sbatch run_sft_torch.sh [N_EXAMPLES]
#   sbatch run_sft_torch.sh 128    # small test
#   sbatch run_sft_torch.sh 512
#   sbatch run_sft_torch.sh        # full dataset

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
CONDA_ENV="${SCRATCH}/conda_envs/a3"
WORK_DIR="${SCRATCH}/transformer-lm-from-scratch/zeroshot_SFT_GRPO"

N_EXAMPLES=${1:-}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export HF_HOME="${SCRATCH}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

cd "${WORK_DIR}"
mkdir -p logs

echo "============================================"
echo "  Part 4 — SFT on MATH dataset"
echo "  N_EXAMPLES: ${N_EXAMPLES:-full}"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: $(python --version)"
echo "============================================"

# Build optional --n-examples flag
N_EXAMPLES_FLAG=""
if [ -n "${N_EXAMPLES}" ]; then
    N_EXAMPLES_FLAG="--n-examples ${N_EXAMPLES}"
fi

python -m student.sft_experiment \
    --model Qwen/Qwen2.5-Math-1.5B \
    --data-path data-distrib/intellect_math/train \
    $N_EXAMPLES_FLAG \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --lr 2e-5 \
    --n-steps 200 \
    --eval-every 50 \
    --max-eval-examples 200 \
    --gpu-memory-utilization 0.85 \
    --output-dir ${SCRATCH}/sft-model-${N_EXAMPLES:-full} \
    --wandb-project nyu-llm-reasoners-a3 \
    --wandb-name sft-n${N_EXAMPLES:-full} \
    --wandb-mode disabled

echo "============================================"
echo "  Done! $(date)"
echo "============================================"
