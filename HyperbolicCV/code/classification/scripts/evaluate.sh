#!/bin/bash

#SBATCH --partition=gpu
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

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


## hyperbolic:
# python classification/test.py -c classification/config/L-CNN.txt \
#    --mode test_accuracy --load_checkpoint classification/output/final_L-CNN_CIFAR-10.pth \
#    --output_dir test_result --dataset CIFAR-10_rot

# still need to do retraining for Lcnn on cifar_10 and cifar_100

# python classification/test.py -c classification/config/L-ResNet18.txt \
#    --mode test_accuracy --load_checkpoint classification/output/final_L-ResNet18_CIFAR-10.pth \
#    --output_dir test_result --dataset CIFAR-10_rot

# python classification/test.py -c classification/config/L-ResNet18.txt \
#    --mode test_accuracy --load_checkpoint classification/output/cifar100_200ep_lorenz/final_L-ResNet18.pth \
#    --output_dir test_result --dataset CIFAR-100_rot


## equivariant
# python classification/test.py -c classification/config/EQE-ResNet18.txt \
#    --mode test_accuracy --load_checkpoint classification/output/EQE-ResNet18_CIFAR-10/final_model.pth \
#    --output_dir test_result --dataset CIFAR-10_rot

# python classification/test.py -c classification/config/EQE-ResNet18.txt \
#    --mode test_accuracy --load_checkpoint classification/output/EQE-ResNet18_CIFAR-100/final_model.pth \
#    --output_dir test_result --dataset CIFAR-100_rot

# python classification/test.py -c classification/config/EQE-CNN.txt \
#    --mode test_accuracy --load_checkpoint classification/output/EQE-CNN_CIFAR-100/final_model.pth \
#    --output_dir test_result --dataset CIFAR-100_rot

# python classification/test.py -c classification/config/EQE-CNN.txt \
#    --mode test_accuracy --load_checkpoint classification/output/EQE-CNN_CIFAR-10/final_model.pth \
#    --output_dir test_result --dataset CIFAR-10_rot


## euclidean
# python classification/test.py -c classification/config/E-ResNet18.txt \
#    --mode test_accuracy --load_checkpoint classification/output/final_E-ResNet18_CIFAR-10.pth \
#    --output_dir test_result --dataset CIFAR-10_rot

# python classification/test.py -c classification/config/E-ResNet18.txt \
#    --mode test_accuracy --load_checkpoint classification/output/final_E-ResNet18_CIFAR-100.pth \
#    --output_dir test_result --dataset CIFAR-100_rot

# python classification/test.py -c classification/config/E-CNN.txt \
#    --mode test_accuracy --load_checkpoint classification/output/final_E-CNN_CIFAR-100.pth \
#    --output_dir test_result --dataset CIFAR-100_rot

# python classification/test.py -c classification/config/E-CNN.txt \
#    --mode test_accuracy --load_checkpoint classification/output/final_E-CNN_CIFAR-10.pth \
#    --output_dir test_result --dataset CIFAR-10_rot

# regular

python classification/test.py -c classification/config/E-CNN.txt \
   --mode test_accuracy --load_checkpoint classification/output/final_E-CNN_MNIST_rotation.pth \
   --output_dir test_result --dataset MNIST