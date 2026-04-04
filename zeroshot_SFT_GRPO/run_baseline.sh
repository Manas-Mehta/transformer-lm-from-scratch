#!/bin/bash
#SBATCH --job-name=math_baseline
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%j_baseline.out
#SBATCH --error=./logs/%j_baseline.err
#SBATCH --requeue

mkdir -p logs

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    set -euo pipefail

    cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO

    # Use HPC pyproject (has vllm) instead of the mac-only default
    cp pyproject-hpc.toml pyproject.toml
    cp uv-hpc.lock uv.lock

    echo "============================================"
    echo "  Part 3 — Zero-Shot MATH Baseline"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"

    uv run python -m student.evaluate \
        --model Qwen/Qwen2.5-Math-1.5B \
        --max-examples 500 \
        --gpu-memory-utilization 0.85 \
        --skip-intellect \
        --verbose

    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
  '
