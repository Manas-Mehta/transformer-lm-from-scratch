#!/bin/bash
#SBATCH --job-name=hw4p1_q1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/q1_%j.out
#SBATCH --error=slurm/logs/q1_%j.err

set -euo pipefail
mkdir -p slurm/logs

# Activate conda env (adjust if your env name differs).
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate nlp_hw4

cd "$SLURM_SUBMIT_DIR"

# Q1: full fine-tune on IMDB + eval on original test set.
# Produces ./out/ and out_original.txt. Target: >= 91% accuracy.
python3 main.py --train --eval
