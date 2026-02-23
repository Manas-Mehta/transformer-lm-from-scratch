#!/bin/bash
#SBATCH --job-name=bench_1_1_4d
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_1_1_4d.out
#SBATCH --error=logs/%j_1_1_4d.err
#SBATCH --requeue

# ─── 1.1.4(d): Full training step with AdamW optimizer ───
# Profiles forward + backward + optimizer.step() to compare
# matmul fraction vs forward-only

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH

    NSYS_DIR=$(dirname $(find ~/tools/nsight-systems -type f -name nsys -perm -111 2>/dev/null | head -n1) 2>/dev/null)
    export PATH=$NSYS_DIR:$PATH

    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism
    mkdir -p profiles

    echo "============================================"
    echo "  1.1.4(d) — Full Training Step Profiling"
    echo "  (forward + backward + AdamW optimizer)"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"

    # Profile full training step WITH optimizer for representative configs
    for SIZE in small medium large xl 2.7B; do
        for CTX in 128 256 512; do
            echo ""
            echo "====== $SIZE, ctx=$CTX, mode=both + optimizer ======"
            nsys profile \
                --trace cuda,nvtx \
                --stats=true \
                --output "profiles/train_${SIZE}_${CTX}" \
                --force-overwrite=true \
                uv run python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode both \
                --optimizer \
                --warmup_steps 3 \
                --measure_steps 1 \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
