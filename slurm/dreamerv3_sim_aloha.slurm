#!/bin/bash
#SBATCH --job-name="dreamerv3_sim_aloha"
#SBATCH --output="/projects/bcyd/ywang41/dreamerv3-torch/slurm/outputs/%x/%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcyd-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

export WANDB_API_KEY=65a3cb43423c6ede326b59f04363f333b5ef5f87
export WANDB_DIR='/projects/bcyd/ywang41/dreamerv3-torch/wandb'
export WANDB_DATA_DIR='/projects/bcyd/ywang41/dreamerv3-torch/wandb/share'
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
conda activate diffusion-forcing
cd /projects/bcyd/ywang41/dreamerv3-torch
python dyn_train.py
