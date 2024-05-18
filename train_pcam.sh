#!/bin/bash

# Arrays of parameters
learning_rates=(3e-3)
epochs=(5)
clipping_norms=(1.0)
epsilons=(1.0)

# Loop over arrays
for lr in "${learning_rates[@]}"; do
    for ep in "${epochs[@]}"; do
        for cn in "${clipping_norms[@]}"; do
            for eps in "${epsilons[@]}"; do
                echo "Training model with lr=$lr, epochs=$ep, clipping norm=$cn, epsilon=$eps"
                # Here you can call your training script with the parameters
		python pcam_training.py --lr $lr --epochs $ep --clip $cn --eps $eps
            done
        done
    done
done
