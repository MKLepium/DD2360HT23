#!/bin/bash

# Define the matrix sizes
SIZES=(
    "1024 1024 1024"
    "2048 1024 1024"
    "1024 2048 1024"
    "2048 2048 1024"
    "2048 2048 2048"
    "3072 2048 2048"
    "3072 3072 2048"
    "3072 3072 3072"
    "4096 3072 3072"
    "4096 4096 3072"
)

# Path to the CUDA executable
EXECUTABLE=./hello_cuda.out

# Check if the executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable $EXECUTABLE does not exist."
    exit 1
fi

# Loop through the sizes and run the program
for SIZE in "${SIZES[@]}"; do
    echo "Running test with size: $SIZE"
    $EXECUTABLE $SIZE
    echo "Test completed."
    echo
done
