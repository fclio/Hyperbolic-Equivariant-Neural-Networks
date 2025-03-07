#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=80:00:00
#SBATCH --output=slurm_output/train_resnet_L-resnet_mnist_rot_%A.out

module purge
module load 2024
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# seem have no use

# pip install -r requirements.txt
# seem like without requirement.txt still runable

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# python classification/train.py -c classification/config/E-ResNet18.txt \
#    --output_dir classification/output --device cuda:0 --dataset CIFAR-10 --num_epochs 200

python classification/train.py -c classification/config/L-ResNet18.txt \
   --output_dir classification/output --device cuda:0 --dataset MNIST_rot --num_epochs 200