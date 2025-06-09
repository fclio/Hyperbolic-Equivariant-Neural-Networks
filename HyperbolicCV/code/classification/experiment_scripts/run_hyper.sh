#!/bin/bash

# Hyperparameter lists
# datasets=( "MNIST" )
# optimizers=( "SGD" "RiemannianSGD" )

# datasets=( "CIFAR-10" "CIFAR-100" "CIFAR-10_rot" "CIFAR-100_rot" )
# optimizers=( "Adam" "RiemannianAdam" )
# learning_rates=( "1e-2" "1e-3" "1e-4" "1e-5" )
batch_sizes=("64" "512")
# num_epochs=("300")
# weight_decay=("1e-4" "1e-3" "1e-5")
num_layers=("50")
embedding_dim=("128" "1024")
# curvature_values=("2.0" "0.5" "0.2")


# Function to submit jobs for each hyperparameter
submit_experiment() {
    param_name=$1
    values=(${!2})
    
    for value in "${values[@]}"; do
        echo "Submitting experiment: $param_name = $value"
        sbatch classification/experiment_scripts/experiment_LEQE.sh --param "$param_name" --value "$value"
    done
}

# Loop through each hyperparameter separately

# submit_experiment "dataset" datasets[@]
# submit_experiment "lr" learning_rates[@]
# submit_experiment "optimizer" optimizers[@]
# submit_experiment "weight_decay" weight_decay[@]
# submit_experiment "num_layers" num_layers[@]
# submit_experiment "embedding_dim" embedding_dim[@]
# submit_experiment "curvature_value" curvature_values[@]
submit_experiment "batch_size" batch_sizes[@]
# submit_experiment "num_epochs" num_epochs[@]
