#!/bin/bash

# iteration
for ((i=1; i<=100; i++)); do
    echo "Iteration $i"

    # Execute the first Python file
    python DeepTarg_gnn.py

    # Execute the second Python file
    python model_deployment.py

    echo "Iteration $i completed"
done
