#!/bin/bash
#SBATCH --job-name=bench_1_1_5c
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_1_1_5c.out
#SBATCH --error=logs/%j_1_1_5c.err
#SBATCH --requeue

# ─── 1.1.5(c): Mixed Precision Benchmarking ───
# Compare FP32 vs BF16 for all 5 model sizes (forward + backward)
# Also runs mixed_precision_test.py and toymodel_mixed_precision.py

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.1.5 — Mixed Precision"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    # ─── mixed_precision_accumulation (5 pts) ───
    echo ">>> mixed_precision_accumulation <<<"
    echo "==================================="
    uv run python student/mixed_precision_test.py
    echo ""

    # ─── (a) ToyModel dtype inspection ───
    echo ">>> ToyModel dtype inspection <<<"
    echo "================================="
    uv run python student/toymodel_mixed_precision.py
    echo ""

    # ─── (c) Benchmark FP32 vs BF16 for all model sizes ───
    echo ">>> FP32 vs BF16 benchmark comparison <<<"
    echo "=========================================="
    for SIZE in small medium large xl 2.7B; do
        echo ""
        echo "=== $SIZE — FP32 ==="
        uv run python student/benchmark.py \
            --model_size "$SIZE" \
            --context_length 128 \
            --mode both \
            --warmup_steps 5 \
            --measure_steps 10

        echo ""
        echo "=== $SIZE — BF16 ==="
        uv run python student/benchmark.py \
            --model_size "$SIZE" \
            --context_length 128 \
            --mode both \
            --warmup_steps 5 \
            --measure_steps 10 \
            --mixed_precision
    done

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
