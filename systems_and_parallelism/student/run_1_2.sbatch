#!/bin/bash
#SBATCH --job-name=bench_1_2
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_1_2.out
#SBATCH --error=logs/%j_1_2.err
#SBATCH --requeue

# ─── 1.2: Benchmarking PyTorch Attention + torch.compile ───

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.2 — Attention Benchmarking"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    # ─── 1.2.1(a) Vanilla attention ───
    echo ">>> 1.2.1(a) Vanilla PyTorch Attention <<<"
    echo "==========================================="
    uv run python student/benchmark_attention.py
    echo ""

    # ─── 1.2.2(a) Compiled attention ───
    echo ">>> 1.2.2(a) Compiled PyTorch Attention <<<"
    echo "============================================"
    uv run python student/benchmark_attention.py --compiled
    echo ""

    # ─── 1.2.2(b) Compiled full model ───
    echo ">>> 1.2.2(b) Compiled Full Model <<<"
    echo "======================================"
    uv run python student/benchmark_attention.py --full_model
    echo ""

    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
