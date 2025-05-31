#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --output=slurm_output/train_LEQECNN_MNIST_%A.out

module purge
module load 2024

# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# seem have no use

# pip install -r requirements.txt
# # seem like without requirement.txt still runable
# pip install datasets==2.14.6 --user

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# python -m pip install xxhash datasets --user

python classification/train.py -c classification/config/EQE-CNN.txt \
   --output_dir classification/output --device cuda:0 --dataset CUB-200 --num_epochs 200 --equivariant_type P4

# python classification/train.py -c classification/config/E-CNN.txt \
#    --output_dir classification/output --device cuda:0 --dataset CIFAR-100 --num_epochs 50 --equivariant_type P4
