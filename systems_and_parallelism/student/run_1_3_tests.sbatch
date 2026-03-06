#!/bin/bash
#SBATCH --job-name=test_1_3
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_1_3_tests.out
#SBATCH --error=logs/%j_1_3_tests.err
#SBATCH --requeue

# ─── 1.3: Run FlashAttention-2 tests ───

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    echo "============================================"
    echo "  1.3 — FlashAttention-2 Tests"
    echo "  Date: $(date)"
    echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "============================================"
    echo ""

    # Run all flash attention tests
    echo ">>> Running all flash attention tests <<<"
    uv run pytest tests/test_attention.py -v
    echo ""

    echo "============================================"
    echo "  Done! $(date)"
    echo "============================================"
'
