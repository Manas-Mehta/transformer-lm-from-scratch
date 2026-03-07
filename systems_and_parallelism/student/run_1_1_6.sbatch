#!/bin/bash
#SBATCH --job-name=bench_1_1_6
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j_1_1_6.out
#SBATCH --error=logs/%j_1_1_6.err
#SBATCH --requeue

# 1.1.6: Memory Profiling — 2.7B model
# Fixed: optimizer only created when mode=both (was always created, causing OOM)
# (a) Two pickle files for memory_viz: forward-only + full training (ctx=128)
# (b) Peak memory per context length (128, 256, 512) for forward + full training
# (c) Same as (b) but with BF16 mixed precision

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.1.6 — Memory Profiling (2.7B)"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"

    # (a)+(b): FP32 forward for ctx=128, 256, 512
    for CTX in 128 256 512; do
        echo ""
        echo "====== 2.7B forward, ctx=$CTX, FP32 ======"
        uv run python student/benchmark.py \
            --model_size 2.7B --context_length $CTX \
            --mode forward --profile_memory \
            --warmup_steps 3 \
        || echo "FAILED (likely OOM) — skipping"
    done

    # (a)+(b): FP32 full training step for ctx=128, 256, 512
    for CTX in 128 256 512; do
        echo ""
        echo "====== 2.7B both, ctx=$CTX, FP32 ======"
        uv run python student/benchmark.py \
            --model_size 2.7B --context_length $CTX \
            --mode both --profile_memory \
            --warmup_steps 3 \
        || echo "FAILED (likely OOM) — skipping"
    done

    # (c): BF16 mixed precision
    for CTX in 128 256 512; do
        echo ""
        echo "====== 2.7B forward, ctx=$CTX, BF16 ======"
        uv run python student/benchmark.py \
            --model_size 2.7B --context_length $CTX \
            --mode forward --profile_memory --mixed_precision \
            --warmup_steps 3 \
        || echo "FAILED (likely OOM) — skipping"
    done

    for CTX in 128 256 512; do
        echo ""
        echo "====== 2.7B both, ctx=$CTX, BF16 ======"
        uv run python student/benchmark.py \
            --model_size 2.7B --context_length $CTX \
            --mode both --profile_memory --mixed_precision \
            --warmup_steps 3 \
        || echo "FAILED (likely OOM) — skipping"
    done

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "  Pickle files in current dir (memory_*.pickle)"
    echo "  Visualize at: https://pytorch.org/memory_viz"
    echo "============================================"
'
