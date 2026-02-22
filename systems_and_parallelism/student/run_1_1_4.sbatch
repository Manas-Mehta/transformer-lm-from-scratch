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
# Profile all model sizes × context lengths with nsys
# Generates .nsys-rep files + text stats for each combo
# Large models with context 1024 may OOM — that's expected, script continues

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.1.4 — Nsight Systems Profiling"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    mkdir -p profiles

    # ─── (a)(b)(c): Profile forward pass for all model sizes ───
    echo ">>> FORWARD PASS PROFILES <<<"
    for SIZE in small medium large xl 2.7B; do
        for CTX in 128 256 512 1024; do
            echo ""
            echo "--- Profiling: $SIZE, ctx=$CTX, mode=forward ---"
            uv run nsys profile \
                -o "profiles/fwd_${SIZE}_${CTX}" \
                --force-overwrite true \
                python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode forward \
                --warmup_steps 5 \
                --measure_steps 1 \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    # ─── (d): Profile full training step (forward + backward) ───
    echo ""
    echo ""
    echo ">>> FULL TRAINING STEP PROFILES (forward+backward) <<<"
    for SIZE in small medium large xl 2.7B; do
        for CTX in 128 256 512 1024; do
            echo ""
            echo "--- Profiling: $SIZE, ctx=$CTX, mode=both ---"
            uv run nsys profile \
                -o "profiles/both_${SIZE}_${CTX}" \
                --force-overwrite true \
                python student/benchmark.py \
                --model_size "$SIZE" \
                --context_length "$CTX" \
                --mode both \
                --warmup_steps 5 \
                --measure_steps 1 \
            || echo "FAILED (likely OOM) — skipping"
        done
    done

    # ─── (e): Profile with NVTX-annotated attention (softmax vs matmul) ───
    echo ""
    echo ""
    echo ">>> NVTX ATTENTION PROFILES (for softmax vs matmul comparison) <<<"
    for SIZE in small medium; do
        for CTX in 128 256 512 1024; do
            echo ""
            echo "--- Profiling attention: $SIZE, ctx=$CTX ---"
            uv run nsys profile \
                -o "profiles/attn_${SIZE}_${CTX}" \
                --force-overwrite true \
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

    # ─── Generate text stats for key profiles ───
    echo ""
    echo ""
    echo ">>> GENERATING TEXT STATS <<<"
    for f in profiles/*.nsys-rep; do
        base=$(basename "$f" .nsys-rep)
        echo "--- Stats: $base ---"
        nsys stats "$f" > "profiles/stats_${base}.txt" 2>&1 || echo "stats failed for $f"
    done

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "  Profile files in: profiles/"
    echo "============================================"
'
