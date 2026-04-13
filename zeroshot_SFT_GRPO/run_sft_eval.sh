#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/%j_sft_eval.out
#SBATCH --error=./logs/%j_sft_eval.err
#SBATCH --requeue

# Usage: sbatch run_sft_eval.sh [MODEL_PATH]
# Examples:
#   sbatch run_sft_eval.sh /scratch/mm14444/sft-model-512
#   sbatch run_sft_eval.sh /scratch/mm14444/sft-model-256
#   sbatch run_sft_eval.sh                                  # defaults to sft-model-512

mkdir -p logs

MODEL_PATH="${1:-/scratch/mm14444/sft-model-512}"

# Write inner script to temp file to avoid singularity quoting issues
INNER_SCRIPT=$(mktemp /scratch/mm14444/tmp/sft_eval_inner_XXXX.sh)
cat > "${INNER_SCRIPT}" <<INNEREOF
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=\$HOME/.local/bin:\$PATH
set -eo pipefail

# ── Persistent caches on /scratch ─────────────────────────────────────
export UV_CACHE_DIR=/scratch/mm14444/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=/scratch/mm14444/.cache/huggingface

# ── Repo + HPC pyproject ──────────────────────────────────────────────
cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO
cp pyproject-hpc.toml pyproject.toml

echo "============================================"
echo "  Part 4.2 — SFT Best Model Evaluation"
echo "  Model: ${MODEL_PATH}"
echo "  Date: \$(date)"
echo "  GPU:  \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

# Evaluate on both Intellect test + MATH test (full 500 examples)
uv run python -m student.evaluate \\
    --model ${MODEL_PATH} \\
    --intellect-path data-distrib/intellect_math/test \\
    --max-examples 500 \\
    --gpu-memory-utilization 0.85 \\
    --verbose

echo "============================================"
echo "  Done! \$(date)"
echo "============================================"
INNEREOF
chmod +x "${INNER_SCRIPT}"

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash "${INNER_SCRIPT}"

rm -f "${INNER_SCRIPT}"
