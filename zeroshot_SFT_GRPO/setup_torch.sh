#!/bin/bash
# ============================================================================
# One-time setup for the Torch cluster (H200/L40S)
# Run on the login node: bash setup_torch.sh
#
# Prerequisites: conda available (base env), internet access, /scratch writable
# ============================================================================
set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
REPO_DIR="${SCRATCH}/transformer-lm-from-scratch"
WORK_DIR="${REPO_DIR}/zeroshot_SFT_GRPO"
CONDA_ENV="${SCRATCH}/conda_envs/a3"
HF_CACHE="${SCRATCH}/hf_cache"

echo "=== Step 1: Clone / update repo ==="
if [ -d "${REPO_DIR}/.git" ]; then
    echo "  Repo exists — pulling latest..."
    cd "${REPO_DIR}" && git pull
else
    echo "  Cloning repo..."
    git clone https://github.com/Manas-Mehta/transformer-lm-from-scratch.git "${REPO_DIR}"
fi

echo ""
echo "=== Step 2: Create conda env (Python 3.12) ==="
echo "  (pyproject requires >=3.11,<3.13 — system Python 3.13 won't work)"
eval "$(conda shell.bash hook)"
if [ -d "${CONDA_ENV}" ]; then
    echo "  Env already exists at ${CONDA_ENV}, skipping create."
else
    conda create -p "${CONDA_ENV}" python=3.12 -y
fi
conda activate "${CONDA_ENV}"
echo "  Python: $(python --version)"

echo ""
echo "=== Step 3: Install project + dependencies ==="
cd "${WORK_DIR}"
cp pyproject-hpc.toml pyproject.toml
pip install -e . 2>&1 | tail -5
echo "  Installed $(pip list 2>/dev/null | wc -l) packages."

echo ""
echo "=== Step 4: Verify critical imports ==="
python -c "
import torch, vllm, transformers, datasets, wandb
print(f'  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')
print(f'  vllm {vllm.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  datasets {datasets.__version__}')
"

echo ""
echo "=== Step 5: Pre-download model + eval dataset ==="
export HF_HOME="${HF_CACHE}"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('  Downloading Qwen/Qwen2.5-Math-1.5B...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
print('  Done.')
"
python -c "
from datasets import load_dataset
print('  Downloading hiyouga/math12k...')
load_dataset('hiyouga/math12k')
print('  Done.')
"

echo ""
echo "=== Step 6: Check training data ==="
if [ -f "${WORK_DIR}/data-distrib/intellect_math/train/data-00000-of-00001.arrow" ]; then
    echo "  Training data found."
else
    echo "  WARNING: Training data not found!"
    echo "  You need to copy data-distrib/ from your local machine:"
    echo ""
    echo "    scp -r zeroshot_SFT_GRPO/data-distrib/ ${NETID}@login.torch.hpc.nyu.edu:${WORK_DIR}/"
    echo ""
    echo "  Or from Greene:"
    echo "    scp -r ${NETID}@greene.hpc.nyu.edu:${SCRATCH}/transformer-lm-from-scratch/zeroshot_SFT_GRPO/data-distrib/ ${WORK_DIR}/"
    echo ""
fi

echo "============================================"
echo "  Setup complete!"
echo "  Conda env: ${CONDA_ENV}"
echo "  Project:   ${WORK_DIR}"
echo "  HF cache:  ${HF_CACHE}"
echo ""
echo "  Next steps:"
echo "    1. Copy data-distrib/ if not present (see above)"
echo "    2. sbatch run_sft_torch.sh 128   # start small to test"
echo "============================================"
