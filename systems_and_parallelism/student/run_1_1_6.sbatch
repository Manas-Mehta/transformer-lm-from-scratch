#!/bin/bash
#SBATCH --job-name=bench_1_1_6
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_1_1_6.out
#SBATCH --error=logs/%j_1_1_6.err
#SBATCH --requeue

# ─── 1.1.6: Memory Profiling ───
# Profile memory usage of the 2.7B model at various context lengths
# Generates .pickle files for visualization at https://pytorch.org/memory_viz

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.1.6 — Memory Profiling"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    mkdir -p memory_profiles

    # ─── (a) Memory timeline for 2.7B model (forward + full training) ───
    echo ">>> (a) Memory timeline — 2.7B, ctx=128 <<<"
    echo ""

    echo "--- Forward only ---"
    uv run python student/benchmark.py \
        --model_size 2.7B --context_length 128 \
        --mode forward --profile_memory \
        --warmup_steps 5

    echo ""
    echo "--- Full training step (forward + backward) ---"
    uv run python student/benchmark.py \
        --model_size 2.7B --context_length 128 \
        --mode both --profile_memory \
        --warmup_steps 5

    # ─── (b) Peak memory table — 2.7B at ctx 128, 256, 512 ───
    echo ""
    echo ""
    echo ">>> (b) Peak memory table — 2.7B, multiple context lengths <<<"
    for CTX in 128 256 512; do
        echo ""
        echo "=== Context $CTX — Forward ==="
        uv run python student/benchmark.py \
            --model_size 2.7B --context_length $CTX \
            --mode forward --profile_memory \
            --warmup_steps 5

        echo ""
        echo "=== Context $CTX — Full training ==="
        uv run python student/benchmark.py \
            --model_size 2.7B --context_length $CTX \
            --mode both --profile_memory \
            --warmup_steps 5
    done

    # ─── (c) Mixed precision memory — 2.7B ───
    echo ""
    echo ""
    echo ">>> (c) Mixed precision memory — 2.7B, ctx=128 <<<"
    echo ""
    echo "--- Forward + BF16 ---"
    uv run python student/benchmark.py \
        --model_size 2.7B --context_length 128 \
        --mode forward --profile_memory --mixed_precision \
        --warmup_steps 5

    echo ""
    echo "--- Full training + BF16 ---"
    uv run python student/benchmark.py \
        --model_size 2.7B --context_length 128 \
        --mode both --profile_memory --mixed_precision \
        --warmup_steps 5

    # Move pickle files to memory_profiles/
    mv memory_*.pickle memory_profiles/ 2>/dev/null || true

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "  Pickle files in: memory_profiles/"
    echo "  Visualize at: https://pytorch.org/memory_viz"
    echo "============================================"
'
