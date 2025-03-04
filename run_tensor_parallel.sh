#!/bin/bash

# Get number of available GPUs
N_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Set required environment variables for distributed training
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE=$N_GPUS

# Run with tensor parallelism
torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    run_sdxl.py 