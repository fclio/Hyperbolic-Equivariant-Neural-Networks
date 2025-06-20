modes=(
  # "test_accuracy"
  # "visualize_embeddings"
  # "fgsm"
  # "pgd"
  # "test_equivairant"
  "test_openood"
)

configs=(
  "E-CNN"
  "EQE-CNN"
  "L-CNN"
  "LEQE-CNN-3_1"
  "LEQE-CNN-3_4"
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
  "CIFAR-10_rot"
  "CIFAR-100_rot"
  # "MNIST_rot"
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

# Fixed parameters
num_epochs=200
equivariant_type="P4"

# Paths
base_config_dir="/home/cfeng/HyperbolicCV/code/classification/config"
base_checkpoint_dir="/home/cfeng/HyperbolicCV/code/classification/output"

for mode in "${modes[@]}"; do
  for config in "${configs[@]}"; do
    for dataset in "${datasets[@]}"; do

      CONFIG_PATH="${base_config_dir}/${config}.txt"
      CHECKPOINT_PATH="${base_checkpoint_dir}/${config}_${dataset}_epoch:${num_epochs}_${equivariant_type}/final_model.pth"
      OUTPUT_PATH="${base_checkpoint_dir}/${config}_${dataset}_epoch:${num_epochs}_${equivariant_type}"

      echo "Submitting: config=$config, dataset=$dataset, mode=$mode"
      sbatch /home/cfeng/HyperbolicCV/code/classification/experiment_scripts/evaluate.sh \
        --config "$CONFIG_PATH" \
        --mode "$mode" \
        --dataset "$dataset" \
        --load_checkpoint "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_PATH"
        
    done
  done
done