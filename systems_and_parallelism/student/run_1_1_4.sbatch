#!/bin/bash
#SBATCH --job-name=bench_1_1_4
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_1_1_4.out
#SBATCH --error=logs/%j_1_1_4.err
#SBATCH --requeue

# ─── 1.1.4: Nsight Systems Profiling ───
# (a) Forward pass timing: nsys vs timeit
# (b)(c) CUDA kernel breakdown (which kernels dominate)
# (d) Full training step (fwd + bwd + optimizer)
# (e) Softmax vs matmul in attention (--nvtx_attention)

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH

    # Add nsys to PATH (per NSYS_doc.pdf)
    NSYS_DIR=$(dirname $(find ~/tools/nsight-systems -type f -name nsys -perm -111 2>/dev/null | head -n1) 2>/dev/null)
    if [ -n "$NSYS_DIR" ]; then
        export PATH=$NSYS_DIR:$PATH
        echo "nsys found at: $(which nsys)"
        echo "nsys version: $(nsys --version)"
    else
        echo "ERROR: nsys not found! Install per NSYS_doc.pdf first."
        exit 1
    fi

    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism
    mkdir -p profiles

    echo "============================================"
    echo "  1.1.4 — Nsight Systems Profiling"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"

    # ─── (a)(b)(c): Forward pass with CUDA kernel stats ───
    echo ""
    echo ">>> SECTION (a)(b)(c): FORWARD PASS — CUDA KERNEL BREAKDOWN <<<"
    for SIZE in small medium large xl 2.7B; do
        for CTX in 128 256 512 1024; do
            echo ""
            echo "====== $SIZE, ctx=$CTX, mode=forward ======"
            uv run nsys profile \
                --trace cuda,nvtx \
                --stats=true \
                --output "profiles/fwd_${SIZE}_${CTX}" \
                --force-overwrite=true \
                python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode forward \
                --warmup_steps 5 \
                --measure_steps 1 \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    # ─── (d): Full training step with optimizer ───
    echo ""
    echo ""
    echo ">>> SECTION (d): FULL TRAINING STEP (fwd + bwd + optimizer) <<<"
    for SIZE in small medium; do
        for CTX in 128 256 512; do
            echo ""
            echo "====== $SIZE, ctx=$CTX, mode=both + optimizer ======"
            uv run nsys profile \
                --trace cuda,nvtx \
                --stats=true \
                --output "profiles/train_${SIZE}_${CTX}" \
                --force-overwrite=true \
                python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode both \
                --optimizer \
                --warmup_steps 3 \
                --measure_steps 1 \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    # ─── (e): NVTX-annotated attention (softmax vs matmul) ───
    echo ""
    echo ""
    echo ">>> SECTION (e): NVTX ATTENTION — SOFTMAX vs MATMUL <<<"
    for SIZE in small medium; do
        for CTX in 128 256 512 1024; do
            echo ""
            echo "====== attention: $SIZE, ctx=$CTX ======"
            uv run nsys profile \
                --trace cuda,nvtx \
                --stats=true \
                --output "profiles/attn_${SIZE}_${CTX}" \
                --force-overwrite=true \
                python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode forward \
                --warmup_steps 5 \
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
