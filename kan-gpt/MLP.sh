#!/bin/bash

#SBATCH --job-name=NN
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=DGX
#SBATCH --exclusive
#SBATCH --output=MLP-%j.out

date

cd ~/DL/kan-gpt
source ~/miniconda3/bin/activate
conda activate NLP

WANDB_MODE=online  python3 -m kan_gpt.train --architecture MLP --batch_size 128 --dataset webtext --max_iters 25 --num_gpus=8 --save 10000 --eval 10000 --seq_len=64

date