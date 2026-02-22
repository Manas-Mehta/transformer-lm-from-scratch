#!/bin/bash
#SBATCH --job-name=bench_1_1_3c
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_1_1_3c.out
#SBATCH --error=logs/%j_1_1_3c.err
#SBATCH --requeue

# ─── 1.1.3(c): Effect of warmup steps on timing ───
# Compare: 0, 1, 2, and 5 warmup steps
# Using small model, context_length 128, forward+backward

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:rw \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.1.3(c) — Warmup effect on benchmarking"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    for WARMUP in 0 1 2 5; do
        echo ""
        echo "--- Warmup steps: $WARMUP ---"
        uv run python student/benchmark.py \
            --model_size small \
            --context_length 128 \
            --mode both \
            --warmup_steps "$WARMUP" \
            --measure_steps 10
    done

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
