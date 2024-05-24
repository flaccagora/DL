#!/bin/bash

#SBATCH --job-name=NN
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --partition=DGX
#SBATCH --exclusive
#SBATCH --output=MLP-%j.out

date

cd ~/DL/kan-gpt
conda deactivate
source .NLP/bin/activate

WANDB_MODE=online  python3 -m kan_gpt.train --architecture MLP --batch_size 32 --dataset webtext --max_iters 100 --num_gpus=4 --save 10000 --eval 10000

date