#!/bin/bash -l

#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name="fm"

# Activate the virtual environment with all the dependencies
conda activate cfm

# Set path
#export PYTHONPATH="/home/ldr934/minFlowMatching:$PYTHONPATH"
cd /home/ldr934/minFlowMatching
export PYTHONPATH=$(pwd):$PYTHONPATH

# See full error
export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

carbontracker python train_fm.py --run-name fm_v2  --learning-rate 5e-3 --weight-decay 1e-4 --scale 16 --train_option data-space --epochs 25  --snapshots-per-sample 5 --condition-snapshots 4 --snapshots-to-generate 1 


