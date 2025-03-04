#!/bin/bash

# Get number of available GPUs
N_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr ',' '\n' | wc -l)

# Set required environment variables for distributed training
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE=$N_GPUS

# Run with tensor parallelism
torchrun \
    nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --sample=cpu \
        --output "profiling_results/sdxl_profile_$(date +"%Y%m%d_%H%M%S")" \
        --stats=true \
        torchrun \
            --nproc_per_node=$N_GPUS \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            --nnodes=1 \
            --node_rank=0 \
            run_sdxl.pys