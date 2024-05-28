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

python eval.py

date