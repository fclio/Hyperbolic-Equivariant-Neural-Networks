#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=8:00:00
#SBATCH --output=slurm_output/evaluate_%A.out

module purge
module load 2024
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# seem have no use

# pip install -r requirements.txt
# seem like without requirement.txt still runable

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


python classification/test.py -c classification/config/L-CNN.txt \
   --mode test_accuracy --load_checkpoint classification/output/L-CNN_CUB-200_epoch:200_P4/final_model.pth \
   --output_dir test_result --dataset CUB-200