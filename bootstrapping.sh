#!/bin/bash

# Loop 5 times
for ((i=1; i<=100; i++)); do
    echo "Iteration $i"

    # Execute the first Python file
    python loop_DeepT.py

    # Execute the second Python file
    python deployment.py

    echo "Iteration $i completed"
done