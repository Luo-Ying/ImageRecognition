#!/bin/bash
#SBATCH --job-name=ImageRecognition
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCh --time=1-00:00:00
#SBATCH --output=./logs_slurm/%x_%A_%a.out
#SBATCH --error=./logs_slurm/%x_%A_%a.err

srun python3 main.py