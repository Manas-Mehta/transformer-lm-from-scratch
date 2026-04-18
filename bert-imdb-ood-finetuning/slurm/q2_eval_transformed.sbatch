#!/bin/bash
#SBATCH --job-name=hw4p1_q2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/q2_%j.out
#SBATCH --error=slurm/logs/q2_%j.err

set -euo pipefail
mkdir -p slurm/logs

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate nlp_hw4

cd "$SLURM_SUBMIT_DIR"

# Q2: evaluate the Q1 model on the custom-transformed test set.
# Produces out_transformed.txt. Target: accuracy drop > 4pt vs Q1 for full credit.
python3 main.py --eval_transformed
