#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=evaluate_equivariant
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/train_evaluate_equivariant_%A.out

module purge
module load 2024
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# seem have no use

# pip install -r requirements.txt
# seem like without requirement.txt still runable

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python classification/train.py -c classification/config/LEQE-CNN.txt \
   --output_dir classification/output --device cuda:0 --dataset MNIST_rot --num_epochs 10

python classification/test.py -c classification/config/LEQE-CNN.txt \
   --mode test_equivairant --load_checkpoint classification/output/LEQE-CNN_MNIST_rot_epoch:10/final_model.pth\
   --num_epochs 10 \
   --dataset MNIST_rot