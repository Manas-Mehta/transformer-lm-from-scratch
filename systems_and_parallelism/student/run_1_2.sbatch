#!/bin/bash
#SBATCH --job-name=sec_1_2
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_1_2.out
#SBATCH --error=logs/%j_1_2.err
#SBATCH --requeue

# ═══════════════════════════════════════════════════
#  Section 1.2 — Problem (pytorch_attention): 5 pts
#    (a) Benchmark vanilla PyTorch attention
#        - batch=8, d_model=[16,32,64,128],
#          seq_len=[256,1024,4096,8192,16384]
#        - 100 forward passes, 100 backward passes
# ═══════════════════════════════════════════════════

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  Section 1.2 — pytorch_attention (5 pts)"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    echo ">>> 1.2(a) Vanilla PyTorch Attention <<<"
    uv run python student/benchmark_attention.py

    echo ""
    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
