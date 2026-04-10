#!/bin/bash
#SBATCH --job-name=sft_math
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=g4-standard-48
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=./logs/%j_sft.out
#SBATCH --error=./logs/%j_sft.err
#SBATCH --requeue

# Usage: sbatch run_sft_g4.sh [N_EXAMPLES]
#   sbatch run_sft_g4.sh 128
#   sbatch run_sft_g4.sh 512
#   sbatch run_sft_g4.sh        # full dataset
#
# Uses the new NVIDIA RTX Pro 6000 partition (single 48GB GPU).
# Both policy model and vLLM share cuda:0 — no 2-GPU setup needed.

mkdir -p logs

N_EXAMPLES=${1:-}

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    set -eo pipefail

    export UV_CACHE_DIR=/scratch/mm14444/.cache/uv
    export UV_LINK_MODE=copy
    export HF_HOME=/scratch/mm14444/.cache/huggingface
    export TMPDIR=/scratch/mm14444/tmp
    mkdir -p $TMPDIR
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO
    cp pyproject-hpc.toml pyproject.toml
    rm -f uv.lock

    echo "============================================"
    echo "  Part 4 — SFT on MATH (single RTX Pro 6000)"
    echo "  N_EXAMPLES: '"${N_EXAMPLES:-full}"'"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "============================================"

    N_EXAMPLES_FLAG=""
    if [ -n "'"${N_EXAMPLES}"'" ]; then
        N_EXAMPLES_FLAG="--n-examples '"${N_EXAMPLES}"'"
    fi

    uv run python -m student.sft_experiment \
        --model Qwen/Qwen2.5-Math-1.5B \
        --data-path data-distrib/intellect_math/train \
        $N_EXAMPLES_FLAG \
        --single-gpu \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --lr 2e-5 \
        --n-steps 200 \
        --eval-every 50 \
        --max-eval-examples 200 \
        --gpu-memory-utilization 0.4 \
        --output-dir /scratch/mm14444/sft-model-'"${N_EXAMPLES:-full}"' \
        --wandb-mode disabled

    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
  '
