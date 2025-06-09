#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=FGSM
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_output/evaluate_FGSM_%A.out

# module spider python

module load 2024

module load Miniconda3/24.7.1-0
conda init
source ~/.bashrc

conda activate py310


python classification/test.py -c classification/config/LEQE-CNN-2_1.txt \
   --mode fgsm \
   --load_checkpoint classification/output/LEQE-CNN-2_1_CIFAR-10_rot_epoch:200_P4/final_model.pth \
   --dataset CIFAR-10 \
   --output_dir classification/output/LEQE-CNN-2_1_CIFAR-10_rot_epoch:200_P4 \
   --equivariant_type P4

# python classification/test.py -c classification/config/E-CNN.txt \
#    --mode pgd \
#    --load_checkpoint classification/output/E-CNN_CIFAR-100_rot_epoch:200_P4/final_model.pth \
#    --dataset CIFAR-100 \
#    --output_dir classification/output/E-CNN_CIFAR-100_rot_epoch:200_P4 \
#    --equivariant_type P4

# python classification/test.py -c classification/config/L-CNN.txt \
#    --mode pgd \
#    --load_checkpoint classification/output/L-CNN_CIFAR-100_rot_epoch:200_P4/final_model.pth \
#    --dataset CIFAR-100 \
#    --output_dir classification/output/L-CNN_CIFAR-100_rot_epoch:200_P4 \
#    --equivariant_type P4

# python classification/test.py -c classification/config/E-CNN.txt \
#    --mode pgd \
#    --load_checkpoint classification/output/E-CNN_CIFAR-10_rot_epoch:200_P4/final_model.pth \
#    --dataset CIFAR-10 \
#    --output_dir classification/output/E-CNN_CIFAR-10_rot_epoch:200_P4 \
#    --equivariant_type P4