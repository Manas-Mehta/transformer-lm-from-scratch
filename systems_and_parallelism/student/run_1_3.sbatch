#!/bin/bash
#SBATCH --job-name=sec_1_3
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j_1_3.out
#SBATCH --error=logs/%j_1_3.err
#SBATCH --requeue

# ═══════════════════════════════════════════════════════════════
#  Section 1.3 — All subparts
#
#  1.3(a)  torch_compile: compiled attention benchmarks (5 pts)
#  1.3(b)  torch_compile: compiled full model benchmarks
#  1.3.1   Weighted Sum example (no deliverable, skip)
#  1.3.2   flash_forward tests (a,b,c)          — 25 pts
#  1.3.2   flash_backward tests                 — 10 pts
#  1.3.2   flash_benchmarking (a)               — 15 pts
# ═══════════════════════════════════════════════════════════════

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  Section 1.3 — All subparts"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    # ─── 1.3(a) torch_compile: compiled attention ───
    echo ">>> 1.3(a) Compiled PyTorch Attention (torch_compile, 5 pts) <<<"
    echo "---------------------------------------------------------------"
    uv run python student/benchmark_attention.py --compiled
    echo ""

    # ─── 1.3(b) torch_compile: compiled full model ───
    echo ">>> 1.3(b) Compiled Full Model (torch_compile) <<<"
    echo "---------------------------------------------------"
    uv run python student/benchmark_attention.py --full_model
    echo ""

    # ─── 1.3.2 flash_forward + flash_backward tests ───
    echo ">>> 1.3.2 FlashAttention-2 Tests (flash_forward 25 pts + flash_backward 10 pts) <<<"
    echo "------------------------------------------------------------------------------------"
    uv run pytest tests/test_attention.py -v
    echo ""

    # ─── 1.3.2 flash_benchmarking ───
    echo ">>> 1.3.2 Flash Benchmarking (flash_benchmarking, 15 pts) <<<"
    echo "--------------------------------------------------------------"
    uv run python student/flash_benchmarking.py
    echo ""

    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
