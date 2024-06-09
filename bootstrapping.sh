#!/bin/bash

# iteration
for ((i=1; i<=100; i++)); do
    echo "Iteration $i"

    # training
    python training_and_evaluation.py

    # deployment
    python model_deployment.py

    echo "Iteration $i completed"
done
