#!/bin/bash
#SBATCH --job-name=bench_1_1_4_kern
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_1_1_4_kernels.out
#SBATCH --error=logs/%j_1_1_4_kernels.err
#SBATCH --requeue

# ─── 1.1.4 (b)(c)(e): Re-run with CUDA kernel tracing ───
# Previous run only had --trace nvtx, missing kernel-level stats
# This adds cuda to get GPU kernel breakdown

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
    echo "  1.1.4 — CUDA Kernel Profiling (b)(c)(e)"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"

    # ─── (b)(c): Forward pass with CUDA kernel stats ───
    echo ""
    echo ">>> FORWARD PASS — CUDA KERNEL BREAKDOWN <<<"
    for SIZE in small medium large xl 2.7B; do
        for CTX in 128 256 512; do
            echo ""
            echo "====== $SIZE, ctx=$CTX, mode=forward ======"
            nsys profile \
                --trace cuda,nvtx \
                --stats=true \
                --output "profiles/kern_fwd_${SIZE}_${CTX}" \
                --force-overwrite=true \
                uv run python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode forward \
                --warmup_steps 3 \
                --measure_steps 1 \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    # ─── (d): Full training step with kernel stats ───
    echo ""
    echo ""
    echo ">>> FULL TRAINING STEP — CUDA KERNEL BREAKDOWN <<<"
    for SIZE in small medium; do
        for CTX in 128 512; do
            echo ""
            echo "====== $SIZE, ctx=$CTX, mode=both ======"
            nsys profile \
                --trace cuda,nvtx \
                --stats=true \
                --output "profiles/kern_both_${SIZE}_${CTX}" \
                --force-overwrite=true \
                uv run python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode both \
                --warmup_steps 3 \
                --measure_steps 1 \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    # ─── (e): NVTX-annotated attention with kernel stats ───
    echo ""
    echo ""
    echo ">>> NVTX ATTENTION — CUDA KERNEL BREAKDOWN (softmax vs matmul) <<<"
    for SIZE in small medium; do
        for CTX in 128 256 512 1024; do
            echo ""
            echo "====== attention: $SIZE, ctx=$CTX ======"
            nsys profile \
                --trace cuda,nvtx \
                --stats=true \
                --output "profiles/kern_attn_${SIZE}_${CTX}" \
                --force-overwrite=true \
                uv run python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode forward \
                --warmup_steps 3 \
                --measure_steps 1 \
                --nvtx_attention \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
