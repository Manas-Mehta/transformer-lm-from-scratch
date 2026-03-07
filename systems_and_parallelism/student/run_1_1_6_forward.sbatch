#!/bin/bash
#SBATCH --job-name=bench_1_1_6_fwd
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_1_1_6_fwd.out
#SBATCH --error=logs/%j_1_1_6_fwd.err
#SBATCH --requeue

# Medium forward-only memory profile (for the forward image in 1.1.6a)

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    uv run python student/benchmark.py \
        --model_size medium --context_length 128 --batch_size 4 \
        --mode forward --profile_memory \
        --warmup_steps 3

    echo "Done. Pickle: memory_medium_128_forward_fp32.pickle"
'
