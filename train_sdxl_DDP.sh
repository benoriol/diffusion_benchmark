#!/bin/bash

# Default parameters
STEPS=100
BATCH_SIZE=16
MICRO_BATCH_SIZE=2
LEARNING_RATE=1e-5
DATASET_SIZE=1000
IMAGE_SIZE=128
NUM_WORKERS=4
PROFILE=false
PROFILE_OUTPUT="sdxl_profile"

# Help message
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --steps N                Number of training steps (default: $STEPS)"
    echo "  --batch-size N           Batch size for training (default: $BATCH_SIZE)"
    echo "  --micro-batch-size N     Micro batch size for gradient accumulation (default: $MICRO_BATCH_SIZE)"
    echo "  --learning-rate N        Learning rate (default: $LEARNING_RATE)"
    echo "  --dataset-size N         Number of samples in dataset (default: $DATASET_SIZE)"
    echo "  --image-size N           Image size for training (default: $IMAGE_SIZE)"
    echo "  --num-workers N          Number of workers for dataloader (default: $NUM_WORKERS)"
    echo "  --profile                Enable NVIDIA Nsight Systems profiling"
    echo "  --profile-output NAME    Profile output name (default: $PROFILE_OUTPUT)"
    echo "  --help                   Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --dataset-size)
            DATASET_SIZE="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --profile-output)
            PROFILE_OUTPUT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print configuration
echo "Running SDXL training with the following configuration:"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Micro batch size: $MICRO_BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Dataset size: $DATASET_SIZE"
echo "  Image size: $IMAGE_SIZE"
echo "  Number of workers: $NUM_WORKERS"
echo "  Profiling: $PROFILE"
if [ "$PROFILE" = true ]; then
    echo "  Profile output: $PROFILE_OUTPUT"
fi
echo ""

# Common arguments for both profiling and non-profiling runs
PYTHON_ARGS=(
    --steps "$STEPS"
    --batch_size "$BATCH_SIZE"
    --micro_batch_size "$MICRO_BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --dataset_size "$DATASET_SIZE"
    --image_size "$IMAGE_SIZE"
    --num_workers "$NUM_WORKERS"
)

# Add profile flag if profiling is enabled
if [ "$PROFILE" = true ]; then
    PYTHON_ARGS+=(--profile)
fi

# Run the training script with or without profiling
if [ "$PROFILE" = true ]; then
    echo "Starting training with NVIDIA Nsight Systems profiling..."
    nsys profile --stats=true \
        --force-overwrite true \
        --output "$PROFILE_OUTPUT" \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --sample=cpu \
        --stats=true \
        python train_sdxl_DDP.py "${PYTHON_ARGS[@]}"
else
    echo "Starting training..."
    python train_sdxl_DDP.py "${PYTHON_ARGS[@]}"
fi

echo "Training completed!" 