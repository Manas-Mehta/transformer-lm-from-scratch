#!/bin/bash
#SBATCH --job-name=hw4p1_q3_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/q3_train_%j.out
#SBATCH --error=slurm/logs/q3_train_%j.err

set -euo pipefail
mkdir -p slurm/logs

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate nlp_hw4

cd "$SLURM_SUBMIT_DIR"

# Q3: train on augmented data (25k original + 5k transformed), eval on transformed test.
# Produces ./out_augmented/ and out_augmented_transformed.txt.
python3 main.py --train_augmented --eval_transformed
