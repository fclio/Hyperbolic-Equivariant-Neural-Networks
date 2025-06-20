#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=evaluate_equivariant
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm_output/evaluate_equivariant_%A.out

module load 2024

module load Miniconda3/24.7.1-0
conda init
source ~/.bashrc

conda activate py310



# python classification/test.py -c classification/config/LEQE-CNN-2.txt \
#    --mode test_equivairant --load_checkpoint /home/cfeng/HyperbolicCV/code/classification/output/LEQE-CNN-3_4_MNIST_rot_epoch:200_P4/final_model.pth \
#    --num_epochs 200 \
#    --exp_v v3_4 \
#    --dataset MNIST_rot

python classification/test.py -c classification/config/EQE-CNN.txt \
   --mode test_equivairant --load_checkpoint /home/cfeng/HyperbolicCV/code/classification/output/EQE-CNN_MNIST_rot_epoch:200_P4/final_model.pth \
   --num_epochs 200 \
   --dataset MNIST_rot

