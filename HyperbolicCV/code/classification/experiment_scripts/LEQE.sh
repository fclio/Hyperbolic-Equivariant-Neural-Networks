#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=LEQE-training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/train_LEQECNN_%j.out

# Parse input args
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --equivariant_type)
      EQ_TYPE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument $1"
      exit 1
      ;;
  esac
done

# Check required args
if [ -z "$CONFIG" ] || [ -z "$DATASET" ]; then
  echo "Usage: $0 --config <config_file> --dataset <dataset_name> [--device <device>] [--num_epochs <epochs>] [--equivariant_type <type>]"
  exit 1
fi

# Default values
DEVICE=${DEVICE:-cuda:0}
NUM_EPOCHS=${NUM_EPOCHS:-60}
EQ_TYPE=${EQ_TYPE:-P4}

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gdown
pip install h5py

OUTPUT_PATH="/home/cfeng/HyperbolicCV/code/classification/output"



python HyperbolicCV/code/classification/train.py -c "$CONFIG" \
  --output_dir "$OUTPUT_PATH" \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --num_epochs "$NUM_EPOCHS" \
  --equivariant_type "$EQ_TYPE" \
  --load_checkpoint "$CHECKPOINT_PATH"
