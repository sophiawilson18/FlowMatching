#!/bin/bash -l

#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name="rollout"

# Activate the virtual environment with all the dependencies
conda activate cfm

# Set path
cd /home/ldr934/FlowMatching
export PYTHONPATH=$(pwd):$PYTHONPATH

# Set full error
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python rollout.py --run-name scale8  --snapshots-per-sample 34 --snapshots-to-generate 30 --condition-snapshots 4  

