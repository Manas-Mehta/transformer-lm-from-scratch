#!/bin/bash
# ============================================================
# OVERNIGHT EXPERIMENT RUNNER — Section 7 (All Deliverables)
# ============================================================
#
# Usage:
#   cd nyu-llm-reasoners-a1
#   bash run_all_experiments.sh
#
# Expected total runtime: ~12 hours on M-series Mac.
# Safe to leave running overnight — continues on failure.
#
# Output structure:
#   experiments/
#   ├── lr_1e-4/training_log.csv     (LR sweep)
#   ├── lr_3e-4/training_log.csv
#   ├── lr_6e-4/training_log.csv     (copied from base)
#   ├── lr_1e-3/training_log.csv
#   ├── lr_3e-3/training_log.csv     (likely diverges)
#   ├── bs_16/training_log.csv       (batch size)
#   ├── bs_32/training_log.csv       (copied from base)
#   ├── bs_64/training_log.csv
#   ├── bs_128/training_log.csv      (might OOM)
#   ├── ablation_no_rmsnorm/         (layer_norm_ablation)
#   ├── ablation_no_rmsnorm_lowlr/
#   ├── ablation_post_norm/          (pre_norm_ablation)
#   ├── ablation_no_rope/            (no_pos_emb)
#   └── ablation_silu_ffn/           (swiglu_ablation)
#
# ============================================================

set -o pipefail

EXPERIMENTS_DIR="experiments"
MASTER_LOG="$EXPERIMENTS_DIR/master_log.txt"
DEVICE="mps"

mkdir -p "$EXPERIMENTS_DIR"

# ---- Helper function ----
run_experiment() {
    local name="$1"
    local dir="$EXPERIMENTS_DIR/$name"
    shift

    echo "" | tee -a "$MASTER_LOG"
    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "[$name] Starting at $(date)" | tee -a "$MASTER_LOG"
    echo "[$name] Command: uv run student/train.py --device $DEVICE --checkpoint_dir $dir $@" | tee -a "$MASTER_LOG"
    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"

    mkdir -p "$dir"

    local start_time=$(date +%s)

    # Run training, tee output to both console and per-experiment log
    uv run student/train.py \
        --device "$DEVICE" \
        --checkpoint_dir "$dir" \
        --save_every 99999 \
        --log_every 50 \
        "$@" \
        2>&1 | tee "$dir/output.log"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    local minutes=$(( duration / 60 ))

    if [ $exit_code -eq 0 ]; then
        # Extract final val loss from output
        local final_loss=$(grep "Loss:" "$dir/output.log" | tail -1 | awk '{print $2}')
        echo "[$name] COMPLETED in ${minutes}m — final_loss=$final_loss" | tee -a "$MASTER_LOG"
    else
        echo "[$name] FAILED (exit code $exit_code) after ${minutes}m" | tee -a "$MASTER_LOG"
    fi

    echo "" | tee -a "$MASTER_LOG"
    return 0  # Always return 0 so script continues
}

# ============================================================
# START
# ============================================================
echo "============================================================" | tee "$MASTER_LOG"
echo "EXPERIMENT RUNNER — Started at $(date)" | tee -a "$MASTER_LOG"
echo "Device: $DEVICE" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"


# ============================================================
# PHASE 1: LEARNING RATE SWEEP — learning_rate (3 pts)
#
# PDF: "Perform a hyperparameter sweep over the learning rates
#        and report the final losses (or note divergence)"
# Deliverable (a): Learning curves for multiple LRs + search strategy
# Deliverable (b): Edge of stability — at least one divergent run
# Target: val_loss <= 2.00 on M-series
# ============================================================
echo "" | tee -a "$MASTER_LOG"
echo "========== PHASE 1: LEARNING RATE SWEEP (3 pts) ==========" | tee -a "$MASTER_LOG"

# Copy base model results (lr=6e-4, already trained)
if [ -f "checkpoints/training_log.csv" ]; then
    mkdir -p "$EXPERIMENTS_DIR/lr_6e-4"
    cp checkpoints/training_log.csv "$EXPERIMENTS_DIR/lr_6e-4/"
    cp checkpoints/checkpoint_final.pt "$EXPERIMENTS_DIR/lr_6e-4/" 2>/dev/null || true
    echo "[lr_6e-4] Copied from existing base model (val_loss=1.6747)" | tee -a "$MASTER_LOG"
fi

run_experiment "lr_1e-4" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 1e-4 --min_learning_rate 1e-5

run_experiment "lr_3e-4" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 3e-4 --min_learning_rate 3e-5

run_experiment "lr_1e-3" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 1e-3 --min_learning_rate 1e-4

# This one likely diverges — needed for "edge of stability" deliverable
run_experiment "lr_3e-3" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 3e-3 --min_learning_rate 3e-4


# ============================================================
# PHASE 2: BATCH SIZE EXPERIMENT — batch_size_experiment (1 pt)
#
# PDF: "Vary your batch size all the way from 1 to the GPU
#        memory limit. Try at least a few batch sizes in between,
#        including typical sizes like 64 and 128."
# Keep total tokens ~40M constant (adjust steps accordingly).
# Re-optimize LR if necessary (linear scaling rule).
# ============================================================
echo "" | tee -a "$MASTER_LOG"
echo "========== PHASE 2: BATCH SIZE EXPERIMENT (1 pt) ==========" | tee -a "$MASTER_LOG"

# bs=32 already done (base model)
if [ -f "checkpoints/training_log.csv" ]; then
    mkdir -p "$EXPERIMENTS_DIR/bs_32"
    cp checkpoints/training_log.csv "$EXPERIMENTS_DIR/bs_32/"
    echo "[bs_32] Copied from existing base model" | tee -a "$MASTER_LOG"
fi

# bs=16: 16*10000*256 = 40.96M tokens, LR scaled down (linear scaling)
run_experiment "bs_16" \
    --batch_size 16 --total_steps 10000 --eval_every 200 \
    --max_learning_rate 3e-4 --min_learning_rate 3e-5

# bs=64: 64*2500*256 = 40.96M tokens, LR scaled up
run_experiment "bs_64" \
    --batch_size 64 --total_steps 2500 --eval_every 50 \
    --max_learning_rate 1.2e-3 --min_learning_rate 1.2e-4

# bs=128: 128*1250*256 = 40.96M tokens (might OOM on M-series)
run_experiment "bs_128" \
    --batch_size 128 --total_steps 1250 --eval_every 25 \
    --max_learning_rate 2.4e-3 --min_learning_rate 2.4e-4


# ============================================================
# PHASE 3: ABLATIONS (4 pts total)
# ============================================================
echo "" | tee -a "$MASTER_LOG"
echo "========== PHASE 3: ABLATIONS (4 pts) ==========" | tee -a "$MASTER_LOG"

# ---- Ablation 1: layer_norm_ablation (1 pt) ----
# PDF: "Remove all of the RMSNorms from your Transformer and train.
#        What happens at the previous optimal learning rate?
#        Can you get stability by using a lower learning rate?"
# Deliverable: 2 learning curves (original LR + best LR) + commentary

echo "" | tee -a "$MASTER_LOG"
echo "--- Ablation 1: No RMSNorm (layer_norm_ablation, 1 pt) ---" | tee -a "$MASTER_LOG"

# At original LR (likely diverges or very unstable)
run_experiment "ablation_no_rmsnorm" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 6e-4 --min_learning_rate 6e-5 \
    --no_rmsnorm

# At lower LR for stability
run_experiment "ablation_no_rmsnorm_lowlr" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 1e-4 --min_learning_rate 1e-5 \
    --no_rmsnorm


# ---- Ablation 2: pre_norm_ablation (1 pt) ----
# PDF: "Modify your pre-norm Transformer implementation into a
#        post-norm one. Train with the post-norm model."
# Post-norm: z = RMSNorm(x + Attn(x)), y = RMSNorm(z + FFN(z))
# Deliverable: Learning curve for post-norm vs pre-norm

echo "" | tee -a "$MASTER_LOG"
echo "--- Ablation 2: Post-Norm (pre_norm_ablation, 1 pt) ---" | tee -a "$MASTER_LOG"

run_experiment "ablation_post_norm" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 6e-4 --min_learning_rate 6e-5 \
    --post_norm


# ---- Ablation 3: no_pos_emb (1 pt) ----
# PDF: "Modify your Transformer implementation with RoPE to remove
#        the position embedding information entirely."
# Deliverable: Learning curve comparing RoPE vs NoPE

echo "" | tee -a "$MASTER_LOG"
echo "--- Ablation 3: No RoPE (no_pos_emb, 1 pt) ---" | tee -a "$MASTER_LOG"

run_experiment "ablation_no_rope" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 6e-4 --min_learning_rate 6e-5 \
    --no_rope


# ---- Ablation 4: swiglu_ablation (1 pt) ----
# PDF: "FFN_SiLU(x) = W2 SiLU(W1 x)"
#      "set d_ff = 4 * d_model, to approximately match the
#       parameter count of the SwiGLU feed-forward network"
# d_ff = 4 * 512 = 2048 for SiLU (vs 1344 for SwiGLU)
# Deliverable: Learning curve SwiGLU vs SiLU + commentary

echo "" | tee -a "$MASTER_LOG"
echo "--- Ablation 4: SiLU FFN (swiglu_ablation, 1 pt) ---" | tee -a "$MASTER_LOG"

run_experiment "ablation_silu_ffn" \
    --batch_size 32 --total_steps 5000 --eval_every 100 \
    --max_learning_rate 6e-4 --min_learning_rate 6e-5 \
    --d_ff 2048 --use_silu_ffn


# ============================================================
# SUMMARY
# ============================================================
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "ALL EXPERIMENTS COMPLETE — $(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Experiment results in: $EXPERIMENTS_DIR/*/training_log.csv" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Next steps:" | tee -a "$MASTER_LOG"
echo "  1. Plot learning curves:  uv run student/plot_experiments.py" | tee -a "$MASTER_LOG"
echo "  2. Check master_log.txt for any failures" | tee -a "$MASTER_LOG"
echo "  3. Write up results for writeup.pdf" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
