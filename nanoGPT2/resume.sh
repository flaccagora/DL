#!/bin/bash

#SBATCH --job-name=NN
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --partition=DGX
#SBATCH --exclusive
#SBATCH --output=MLP-%j.out

date

cd ~/DL/nanoGPT
source ~/miniconda3/bin/activate
conda activate NLP

torchrun --nproc_per_node=8 --nnodes=1 train.py --architecture='MLP' --init_from='resume' --wandb_run_name='gpt2'

date