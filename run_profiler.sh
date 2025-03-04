#!/bin/bash

# Create a directory for profiling results if it doesn't exist
mkdir -p profiling_results

# Get current timestamp for unique output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="profiling_results/sdxl_profile_${TIMESTAMP}"

# Run the Python script with NVIDIA profiler
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=cpu \
  --output "${OUTPUT_FILE}" \
  --stats=true \
  python run_sdxl.py

echo "Profiling completed. Results saved to: ${OUTPUT_FILE}"
