#!/bin/bash

# Create a directory for profiling results if it doesn't exist
mkdir -p profiling_results

# Get current timestamp for unique output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="profiling_results/sdxl_profile_${TIMESTAMP}"

# Get number of GPUs
N_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

if [ "$N_GPUS" -gt 1 ]; then
    echo "Running with tensor parallelism on $N_GPUS GPUs"
    # Set required environment variables for distributed training
    export MASTER_ADDR="localhost"
    export MASTER_PORT="29500"
    export WORLD_SIZE=$N_GPUS
    
    # Run the Python script with NVIDIA profiler and tensor parallelism
    nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --sample=cpu \
        --output "${OUTPUT_FILE}" \
        --stats=true \
        torchrun \
            --nproc_per_node=$N_GPUS \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            --nnodes=1 \
            --node_rank=0 \
            run_sdxl.py
else
    echo "Running on single GPU"
    # Run the Python script with NVIDIA profiler in single GPU mode
    nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --sample=cpu \
        --output "${OUTPUT_FILE}" \
        --stats=true \
        python run_sdxl.py
fi

echo "Profiling completed. Results saved to: ${OUTPUT_FILE}"
