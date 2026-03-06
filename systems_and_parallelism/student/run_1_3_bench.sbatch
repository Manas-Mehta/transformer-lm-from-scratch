#!/bin/bash
#SBATCH --job-name=bench_flash
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_1_3_bench.out
#SBATCH --error=logs/%j_1_3_bench.err
#SBATCH --requeue

# --- 1.3.2 Problem (flash_benchmarking): Triton FA2 vs vanilla attention ---

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.3.2 — Flash Benchmarking"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    uv run python student/flash_benchmarking.py

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
