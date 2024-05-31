#!/bin/bash

#SBATCH --job-name=NN
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=DGX
#SBATCH --exclusive
#SBATCH --output=MLP-%j.out

date

cd ~/DL/nanoGPT
source ~/miniconda3/bin/activate
conda activate NLP

torchrun --nproc_per_node=8 --nnodes=1 train.py --architecture='MLP' --learning_rate=1e-3 --min_lr=6e-5 --max_iters=12000 --lr_decay_iters=12000

date