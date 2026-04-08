#!/bin/bash
#SBATCH --job-name=sft_math
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=./logs/%j_sft.out
#SBATCH --error=./logs/%j_sft.err
#SBATCH --requeue

# Usage: sbatch run_sft.sh [N_EXAMPLES]
# Examples:
#   sbatch run_sft.sh 128
#   sbatch run_sft.sh 256
#   sbatch run_sft.sh 512
#   sbatch run_sft.sh 1024
# Default (no arg): full dataset

mkdir -p logs

N_EXAMPLES=${1:-}   # empty = use full dataset

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    set -euo pipefail

    # ── Persistent caches on /scratch ─────────────────────────────────────
    export UV_CACHE_DIR=/scratch/mm14444/.cache/uv
    export UV_LINK_MODE=copy
    export HF_HOME=/scratch/mm14444/.cache/huggingface
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # ── Repo ──────────────────────────────────────────────────────────────
    # All deps are pre-installed in .venv. Skip uv sync (avoids flash-attn
    # build failure). Python finds the student package from cwd.
    cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO

    echo "============================================"
    echo "  Part 4 — SFT on MATH dataset"
    echo "  N_EXAMPLES: '"${N_EXAMPLES:-full}"'"
    echo "  Date: $(date)"
    echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr "\n" " ")"
    echo "============================================"

    # Build optional --n-examples flag
    N_EXAMPLES_FLAG=""
    if [ -n "'"${N_EXAMPLES}"'" ]; then
        N_EXAMPLES_FLAG="--n-examples '"${N_EXAMPLES}"'"
    fi

    .venv/bin/python -m student.sft_experiment \
        --model Qwen/Qwen2.5-Math-1.5B \
        --data-path data-distrib/intellect_math/train \
        $N_EXAMPLES_FLAG \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --lr 2e-5 \
        --n-steps 200 \
        --eval-every 50 \
        --max-eval-examples 200 \
        --gpu-memory-utilization 0.85 \
        --output-dir /scratch/mm14444/sft-model-'"${N_EXAMPLES:-full}"' \
        --wandb-project nyu-llm-reasoners-a3 \
        --wandb-name sft-n'"${N_EXAMPLES:-full}"' \
        --wandb-mode disabled

    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
  '
