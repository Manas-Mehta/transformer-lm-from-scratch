#!/bin/bash
# ============================================================================
# One-time login-node script. Pre-caches `google-t5/t5-small` into
# ${SCRATCH}/hf_cache so compute nodes (HF_HUB_OFFLINE=1) can load it.
#
# The conda env from Part 1 (/scratch/mm14444/conda_envs/nlp_hw4) already has
# everything we need: torch 2.x, transformers>=4.35, sentencepiece. We just
# need to download the T5 checkpoint + tokenizer once.
#
# Usage: bash slurm/setup_t5_cache.sh
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
ENV_DIR="${SCRATCH}/conda_envs/nlp_hw4"

export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "${HF_HOME}"

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

# sentencepiece is needed by the T5Tokenizer (slow path fallback) and is
# imported by T5TokenizerFast in some transformers versions. Install if missing.
python -c "import sentencepiece" 2>/dev/null || pip install "sentencepiece>=0.1.99"

echo "=== Pre-caching google-t5/t5-small ==="
python -c "
import os
print('HF_HOME =', os.environ.get('HF_HOME'))
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config
T5TokenizerFast.from_pretrained('google-t5/t5-small')
T5Config.from_pretrained('google-t5/t5-small')
T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
print('=== t5-small cache ready ===')
"
