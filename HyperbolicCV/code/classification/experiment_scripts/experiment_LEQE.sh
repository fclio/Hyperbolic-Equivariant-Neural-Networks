#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=80:00:00
#SBATCH --output=slurm_output/train_%A.out  # Dynamic output filename



module purge
module load 2024

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Default hyperparameter values
dataset="MNIST_rot"
lr="1e-1"
batch_size="128"
optimizer="RiemannianSGD"
num_epochs="200"
weight_decay="5e-4"
num_layers="18"
embedding_dim="512"
curvature_value="1.0"

# Parse command-line arguments
while [[ $# -gt 0 ]]
do
    case "$1" in
        --param)
            param="$2"
            shift 2
            ;;
        --value)
            value="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Update the selected parameter
case "$param" in
    dataset) dataset="$value" ;;
    lr) lr="$value" ;;
    batch_size) batch_size="$value" ;;
    optimizer) optimizer="$value" ;;
    num_epochs) num_epochs="$value" ;;
    weight_decay) weight_decay="$value" ;;
    num_layers) num_layers="$value" ;;
    embedding_dim) embedding_dim="$value" ;;
    curvature_value) curvature_value="$value" ;;
    *)
        echo "Invalid parameter: $param"
        exit 1
        ;;
esac

# Display the configuration
echo "Running experiment with:"
echo "Dataset: $dataset"
echo "Learning Rate: $lr"
echo "Batch Size: $batch_size"
echo "Optimizer: $optimizer"
echo "Epochs: $num_epochs"
echo "Weight Decay: $weight_decay"
echo "Number of Layers: $num_layers"
echo "Embedding Dimension: $embedding_dim"
echo "Curvature Value: $curvature_value"

# Run the training script
python classification/train.py \
    -c classification/config/LEQE-CNN.txt \
    --output_dir classification/output_hyperparameter/${param}/${value}_3 \
    --device cuda:0 \
    --dataset "$dataset" \
    --lr "$lr" \
    --batch_size "$batch_size" \
    --optimizer "$optimizer" \
    --num_epochs "$num_epochs" \
    --weight_decay "$weight_decay" \
    --num_layers "$num_layers" \
    --embedding_dim "$embedding_dim" \
    --encoder_k "$curvature_value" \
    --decoder_k "$curvature_value" \
