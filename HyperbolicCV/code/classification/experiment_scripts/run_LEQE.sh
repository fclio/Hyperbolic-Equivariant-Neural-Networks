#!/bin/bash

# List of CNN sizes to try
cnn_list=(
  # "small" 
  # "big"
  ""
  )

# List of config files (relative paths)
configs=(
  # "E-CNN"
  # "EQE-CNN"
  "L-CNN"
  # "LEQE-CNN-3_1"
  # "LEQE-CNN-3_4"
  # "LEQE-CNN-4_3_1"
  # "LEQE-CNN-4_1_3_1"
  # "LEQE-CNN-4_2_3_1"
  # "LEQE-CNN-4"
  # "LEQE-CNN-4_1"
  # "LEQE-CNN-4_2"
  # "LEQE-CNN-4_3_2"
  # "LEQE-CNN-4_1_3_2"
  # "LEQE-CNN-4_2_3_2"
)

datasets=(
  # "Flower102"
  # "CIFAR-10_rot"
  "CIFAR-100_rot"
  "MNIST_rot"
  # "cifar100-lt"
  # "cifar10-lt"
  # "Tiny-ImageNet"
  # "CUB-200"
  # "Flower102"
  # "Food101"
  # "PCAM"
  # "PET"
  # "SUN397"
  # "DTD"
)

# Other fixed parameters (optional)
device="cuda:0"
num_epochs=200
equivariant_type="P4"

for config in "${configs[@]}"; do
  for dataset in "${datasets[@]}"; do
    for cnn_size in "${cnn_list[@]}"; do

      echo "Submitting job for config=$config, dataset=$dataset, cnn_size=$cnn_size"

      CONFIG_PATH="/home/cfeng/HyperbolicCV/code/classification/config/${config}.txt"
      CHECKPOINT_PATH=""
      # CHECKPOINT_PATH="classification/output/${config}_${dataset}_epoch:${num_epochs}_${equivariant_type}/step_model.pth"

      sbatch /home/cfeng/HyperbolicCV/code/classification/experiment_scripts/LEQE.sh \
        --config "$CONFIG_PATH" \
        --dataset "$dataset" \
        --device "$device" \
        --num_epochs "$num_epochs" \
        --equivariant_type "$equivariant_type" \
        --cnn_size "$cnn_size" \
        --checkpoint "$CHECKPOINT_PATH"
    done
  done
done
