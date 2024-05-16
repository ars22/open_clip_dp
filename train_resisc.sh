#!/bin/bash

# Arrays of parameters
learning_rates=(1e-4 5e-5)
epochs=(8)
clipping_norms=(2.0)
epsilons=(0.3 0.4)

# Loop over arrays
for lr in "${learning_rates[@]}"; do
    for ep in "${epochs[@]}"; do
        for cn in "${clipping_norms[@]}"; do
            for eps in "${epsilons[@]}"; do
                echo "Training model with lr=$lr, epochs=$ep, clipping norm=$cn, epsilon=$eps"
                # Here you can call your training script with the parameters
		python resisc_training.py --lr $lr --epochs $ep --clip $cn --eps $eps
            done
        done
    done
done
