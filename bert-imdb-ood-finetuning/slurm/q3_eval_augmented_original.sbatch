#!/bin/bash
#SBATCH --job-name=hw4p1_q3_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/q3_eval_%j.out
#SBATCH --error=slurm/logs/q3_eval_%j.err

set -euo pipefail
mkdir -p slurm/logs

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate nlp_hw4

cd "$SLURM_SUBMIT_DIR"

# Q3: evaluate augmented model on the original test set.
# Produces out_augmented_original.txt.
python3 main.py --eval --model_dir out_augmented
