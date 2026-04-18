#!/bin/bash
# ============================================================================
# One-time HPC setup. Run this ONCE on the torch login node before any sbatch.
# It creates the conda env and pre-downloads all HF / NLTK assets so compute
# nodes can run fully offline (HF_HUB_OFFLINE=1).
#
# Usage:  bash slurm/setup_env.sh
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/transformer-lm-from-scratch/bert-imdb-ood-finetuning"
ENV_DIR="${SCRATCH}/conda_envs/nlp_hw4"

echo "=== Step 1/4: create conda env at ${ENV_DIR} ==="
eval "$(conda shell.bash hook)"
if [ ! -d "${ENV_DIR}" ]; then
    conda create -p "${ENV_DIR}" python=3.10 -y
fi
conda activate "${ENV_DIR}"

echo "=== Step 2/4: install Python packages (relaxed pins) ==="
# Pinned requirements.txt (torch 1.13.1) lacks wheels for Python 3.10 on some
# CUDA builds. Use relaxed versions that work on H200 (sm_90, CUDA 12).
pip install --upgrade pip
pip install \
    "torch>=2.1,<2.5" \
    "transformers>=4.35,<4.45" \
    "datasets>=2.14,<3.0" \
    "evaluate>=0.4" \
    "scikit-learn>=1.2" \
    "nltk>=3.8" \
    "tqdm" \
    "numpy<2.0"

echo "=== Step 3/4: download NLTK data ==="
python -c "
import nltk
for pkg in ['wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
    nltk.download(pkg, quiet=False)
print('nltk data ok')
"

echo "=== Step 4/4: pre-cache IMDB dataset + bert-base-cased to HF cache ==="
export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "${HF_HOME}"
python -c "
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('HF_HOME =', os.environ.get('HF_HOME'))
ds = load_dataset('imdb')
print('IMDB splits:', list(ds.keys()))
AutoTokenizer.from_pretrained('bert-base-cased')
AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
print('=== setup complete ===')
"

echo ""
echo "Environment ready."
echo "  Env:    ${ENV_DIR}"
echo "  Cache:  ${SCRATCH}/hf_cache"
echo ""
echo "Next: cd ${PROJECT_DIR} && sbatch slurm/q1_train_eval.sbatch"
