#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/evaluate_%A.out

module purge
module load 2024
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# seem have no use

# pip install -r requirements.txt
# seem like without requirement.txt still runable

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python test.py -c classification/config/L-CNN.txt \
   --mode test_accuracy --load_checkpoint classification/output/final_L-CNN_MNIST_rotation.pth \
   --output_dir test_result --dataset MNIST_rotation
  