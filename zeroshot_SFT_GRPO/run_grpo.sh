#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=./logs/%j_grpo.out
#SBATCH --error=./logs/%j_grpo.err
#SBATCH --requeue

# Usage: sbatch run_grpo.sh [LR] [LOSS_TYPE] [EXTRA_ARGS...]
# Examples:
#   sbatch run_grpo.sh                                    # defaults: lr=1e-5, reinforce_with_baseline
#   sbatch run_grpo.sh 5e-6                               # custom lr
#   sbatch run_grpo.sh 1e-5 no_baseline                   # custom loss type
#   sbatch run_grpo.sh 1e-5 reinforce_with_baseline --no-use-std-normalization

mkdir -p logs

LR="${1:-1e-5}"
LOSS_TYPE="${2:-reinforce_with_baseline}"
shift 2 2>/dev/null || true  # remaining args passed through

INNER_SCRIPT=$(mktemp /scratch/mm14444/tmp/grpo_inner_XXXX.sh)
cat > "${INNER_SCRIPT}" <<INNEREOF
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=\$HOME/.local/bin:\$PATH
set -eo pipefail

# ── Persistent caches on /scratch ─────────────────────────────────────
export UV_CACHE_DIR=/scratch/mm14444/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=/scratch/mm14444/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Repo + HPC pyproject ──────────────────────────────────────────────
cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO
cp pyproject-hpc.toml pyproject.toml

echo "============================================"
echo "  Part 7 — GRPO on Countdown"
echo "  LR: ${LR}"
echo "  Loss type: ${LOSS_TYPE}"
echo "  Extra args: $@"
echo "  Date: \$(date)"
echo "  GPUs: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

uv run python -m student.grpo_experiment \\
    --lr ${LR} \\
    --loss-type ${LOSS_TYPE} \\
    --n-grpo-steps 200 \\
    --rollout-batch-size 16 \\
    --group-size 8 \\
    --gradient-accumulation-steps 8 \\
    --eval-every 10 \\
    --n-eval-examples 200 \\
    --gpu-memory-utilization 0.8 \\
    --output-dir /scratch/mm14444/grpo-model-lr${LR}-${LOSS_TYPE} \\
    --wandb-name grpo-lr${LR}-${LOSS_TYPE} \\
    --wandb-mode disabled \\
    $@

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
