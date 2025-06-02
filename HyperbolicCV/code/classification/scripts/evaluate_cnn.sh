#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/evaluate_%A.out

module load 2024

module load Miniconda3/24.7.1-0
conda init
source ~/.bashrc

conda activate py310   

# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# seem have no use

# pip install -r requirements.txt
# seem like without requirement.txt still runable


python classification/test.py -c classification/config/LEQE-CNN-2_1.txt \
   --mode test_accuracy \
   --load_checkpoint classification/output/LEQE-CNN-2_1_Tiny-ImageNet_epoch:200_P4/best_model.pth \
   --num_epochs 10 \
   --dataset Tiny-ImageNet \
   --output_dir classification/output/LEQE-CNN-2_1_Tiny-ImageNet_epoch:200_P4 \
   --P4 \
