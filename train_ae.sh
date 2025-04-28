#!/bin/bash -l

#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=32
#SBATCH --time=26:00:00
#SBATCH --job-name="ae"

# Activate the virtual environment with all the dependencies
conda activate cfm

# Set path
#export PYTHONPATH="/home/ldr934/minFlowMatching:$PYTHONPATH"
cd /home/ldr934/minFlowMatching
export PYTHONPATH=$(pwd):$PYTHONPATH

# See full error
export HYDRA_FULL_ERROR=1

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

export CUDA_VISIBLE_DEVICES=0

carbontracker python train_ae.py --run-name shearflow_ae --ae_learning_rate 1e-3 --ae_loss mse --ae_epochs 24 --snapshots-per-sample 25  
#python train_ae.py --run-name shearflow_ae --ae_learning_rate 1e-4 --ae_loss mse --ae_epochs 30 --snapshots-per-sample 15
#python train_ae.py --run-name shearflow_ae --ae_learning_rate 5e-4 --ae_loss mse --ae_epochs 30 --snapshots-per-sample 25
