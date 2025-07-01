#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=slurm_output/evaluate_%j.out

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Parse input args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"; shift 2;;
    --mode)
      MODE="$2"; shift 2;;
    --load_checkpoint)
      CHECKPOINT="$2"; shift 2;;
    --dataset)
      DATASET="$2"; shift 2;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2;;
    *)
      echo "Unknown argument: $1"; exit 1;;
  esac
done

# --- Validate required args ---
if [[ -z "$CONFIG" || -z "$MODE" || -z "$CHECKPOINT" || -z "$DATASET" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --config <path> --mode <mode> --load_checkpoint <path> --dataset <name> --output_dir <path>"
  exit 1
fi

# --- Environment setup ---
module purge
module load 2024
module load Miniconda3/24.7.1-0
source ~/.bashrc
conda activate py310
pip install h5py
pip install datasets==2.14.5


# --- Run evaluation ---
python HyperbolicCV/code/classification/test.py \
  --config "$CONFIG" \
  --mode "$MODE" \
  --load_checkpoint "$CHECKPOINT" \
  --num_epochs 200 \
  --dataset "$DATASET" \
  --equivariant_type "P4" \
  --output_dir "$OUTPUT_DIR"
