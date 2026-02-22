#!/bin/bash
#SBATCH --job-name=bench_1_1_3b
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_1_1_3b.out
#SBATCH --error=logs/%j_1_1_3b.err
#SBATCH --requeue

# ─── 1.1.3(b): Time forward and forward+backward for all 5 model sizes ───
# Uses 5 warmup steps and 10 measurement steps (defaults in benchmark.py)
# Context length fixed at 128, batch size 4

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:rw \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.1.3(b) — Benchmarking all model sizes"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    echo ">>> FORWARD PASS ONLY <<<"
    echo "========================="
    for SIZE in small medium large xl 2.7B; do
        echo ""
        echo "--- Model: $SIZE, Mode: forward ---"
        uv run python student/benchmark.py \
            --model_size "$SIZE" \
            --context_length 128 \
            --mode forward \
            --warmup_steps 5 \
            --measure_steps 10
    done

    echo ""
    echo ""
    echo ">>> FORWARD + BACKWARD <<<"
    echo "=========================="
    for SIZE in small medium large xl 2.7B; do
        echo ""
        echo "--- Model: $SIZE, Mode: both ---"
        uv run python student/benchmark.py \
            --model_size "$SIZE" \
            --context_length 128 \
            --mode both \
            --warmup_steps 5 \
            --measure_steps 10
    done

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
