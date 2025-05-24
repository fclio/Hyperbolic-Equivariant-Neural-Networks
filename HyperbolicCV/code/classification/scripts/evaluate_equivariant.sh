#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=evaluate_equivariant
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/evaluate_equivariant_%A.out

module purge
module load 2024
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# seem have no use

# pip install -r requirements.txt
# seem like without requirement.txt still runable

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python classification/test.py -c classification/config/E-CNN.txt \
   --mode test_equivairant --load_checkpoint classification/output/E-CNN_MNIST/final_model.pth \
   --dataset MNIST