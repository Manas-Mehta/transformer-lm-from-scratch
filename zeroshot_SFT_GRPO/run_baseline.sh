#!/bin/bash
#SBATCH --job-name=math_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/baseline_%j.out

mkdir -p logs

cd $SCRATCH/transformer-lm-from-scratch/zeroshot_SFT_GRPO

uv run python -m student.evaluate \
    --model Qwen/Qwen2.5-Math-1.5B \
    --max-examples 500 \
    --gpu-memory-utilization 0.85 \
    --verbose
