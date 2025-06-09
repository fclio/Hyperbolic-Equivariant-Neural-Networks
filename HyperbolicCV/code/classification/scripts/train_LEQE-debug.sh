#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00
#SBATCH --output=slurm_output/train_debug_%A.out

module purge
module load 2023

module load Python/3.11.3-GCCcore-12.3.0

pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install gdown

python classification/train_test.py -c classification/config/LEQE-CNN-2_1.txt \
   --output_dir classification/output --device cuda:0 --dataset cifar100-lt --num_epochs 175 --equivariant_type P4 \
   --load_checkpoint classification/output/LEQE-CNN-2_1_cifar100-lt_epoch:200_P4_error/step_model.pth


# python classification/train.py -c classification/config/E-CNN.txt \
#    --output_dir classification/output --device cuda:0 --dataset CIFAR-100 --num_epochs 50 --equivariant_type P4
