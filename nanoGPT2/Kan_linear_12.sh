#!/bin/bash

#SBATCH --job-name=NN
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --partition=DGX
#SBATCH --exclusive
#SBATCH --output=KAN-%j.out

date

cd ~/DL/nanoGPT
source ~/miniconda3/bin/activate
conda activate NLP

torchrun --nproc_per_node=8 --nnodes=1 \
                    train.py --architecture='KAN' --learning_rate=1e-3 --min_lr=6e-5 \
                    --weight_decay=.0 --dropout=.0 --wandb_run_name='KAN-Linear-GPT2-12' \
                    --n_layer=12 --n_embd=360 \
                    --k=3 --grid=3 \
                    --max_iters=12000 --lr_decay_iters=12000 \
                    --out_dir='out_lin_kan_12' \
                    --attn='Linear_Attn'   

date