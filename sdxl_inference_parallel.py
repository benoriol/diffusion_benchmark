from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch
import time
import torch.cuda.nvtx as nvtx
import os
from PIL import Image
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run SDXL with profiling')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for image generation')
args = parser.parse_args()

# Check if we're running in distributed mode
is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
# Get parallelism mode from environment variable (FSDP or DDP)
parallel_mode = os.environ.get("PARALLEL_MODE", "FSDP").upper()

if is_distributed:
    # Import necessary modules based on parallelism mode
    import torch.distributed as dist
    
    if parallel_mode == "FSDP":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            BackwardPrefetch,
            ShardingStrategy,
            CPUOffload
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
    elif parallel_mode == "DDP":
        from torch.nn.parallel import DistributedDataParallel as DDP

# Configuration
N_WARMUP = int(os.environ.get("N_WARMUP", "0"))  # Number of warmup iterations
N_ITERATIONS = int(os.environ.get("N_ITERATIONS", "2"))  # Number of measured iterations
# Get batch size from environment variable or fallback to command line arg
BATCH_SIZE = int(os.environ.get("SDXL_BATCH_SIZE", args.batch_size))
VAE_BATCH_SIZE = min(int(os.environ.get("VAE_BATCH_SIZE", "4")), BATCH_SIZE)  # Batch size for VAE decoding to avoid OOM
prompt = os.environ.get("PROMPT", "A majestic mountain landscape at sunset")
prompts = [prompt] * BATCH_SIZE  # Replicate prompt for batch processing

def setup_tensor_parallel():
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        return torch.device(f"cuda:{dist.get_rank()}")
    return torch.device("cuda")

def decode_latents_in_batches(pipe, latents, device):
    """Decode latents in smaller batches to avoid OOM"""
    all_images = []
    for i in range(0, latents.shape[0], VAE_BATCH_SIZE):
        batch_latents = latents[i:i + VAE_BATCH_SIZE].to(device)
        # Scale and decode the image latents with vae
        batch_latents = 1 / pipe.vae.config.scaling_factor * batch_latents
        with torch.inference_mode():
            batch_images = pipe.vae.decode(batch_latents).sample
        batch_images = (batch_images / 2 + 0.5).clamp(0, 1)
        # Convert to PIL images
        batch_images = batch_images.cpu().permute(0, 2, 3, 1).float().numpy()
        batch_images = (batch_images * 255).round().astype("uint8")
        batch_pil_images = [Image.fromarray(image) for image in batch_images]
        all_images.extend(batch_pil_images)
        torch.cuda.empty_cache()  # Clear GPU memory after each batch
    return all_images

def generate_images(pipe, prompts, device):
    """Generate images with batched VAE decoding"""
    with torch.inference_mode():
        # Run text encoder and unet
        outputs = pipe(
            prompts,
            output_type="latent",
        )
        # Ensure latents are on the correct device
        latents = outputs.images.to(device)
        # Decode latents in batches
        images = decode_latents_in_batches(pipe, latents, device)
    return images

def create_pipeline():
    if is_distributed:
        print(f"Loading model with {parallel_mode} parallelism...")
    else:
        print("Loading model in single GPU mode...")
    
    print(f"Using batch size: {BATCH_SIZE} (VAE batch size: {VAE_BATCH_SIZE})")
    
    # Initialize device
    device = setup_tensor_parallel()
    
    # Load base model
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    if is_distributed:
        if parallel_mode == "FSDP":
            # Configure FSDP settings
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
            
            # Create size-based policy with fixed arguments
            wrapping_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=1_000_000,
                recurse=True,
                nonwrapped_numel=1_000_000
            )
            
            # Wrap UNet with FSDP
            pipe.unet = FSDP(
                pipe.unet,
                auto_wrap_policy=wrapping_policy,
                mixed_precision=mp_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=torch.cuda.current_device(),
                cpu_offload=CPUOffload(offload_params=False)
            )
        elif parallel_mode == "DDP":
            # Move UNet to device first
            pipe.unet = pipe.unet.to(device)
            # Wrap UNet with DDP
            pipe.unet = DDP(
                pipe.unet,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                broadcast_buffers=False,
                find_unused_parameters=False
            )
        
        # Move other components to device
        pipe.text_encoder = pipe.text_encoder.to(device)
        pipe.text_encoder_2 = pipe.text_encoder_2.to(device)
        pipe.vae = pipe.vae.to(device)
        
        print("\nModel distribution:")
        print(f"Parallelism mode: {parallel_mode}")
        print(f"World size: {dist.get_world_size()}")
        print(f"Current rank: {dist.get_rank()}")
        print(f"Device: {device}")
    else:
        # Single GPU mode
        pipe = pipe.to(device)
        print(f"\nRunning on device: {device}")
    
    return pipe, device

# Initialize pipeline
pipe, device = create_pipeline()

# Clear CUDA cache
torch.cuda.empty_cache()

# Create output directory for batch images
os.makedirs("outputs", exist_ok=True)

# Warmup runs
print("\nPerforming warmup iterations...")
for i in range(N_WARMUP):
    with torch.inference_mode():
        nvtx.range_push(f"Warmup_{i}")
        images = generate_images(pipe, prompts, device)
        nvtx.range_pop()
    print(f"Warmup iteration {i+1}/{N_WARMUP} complete")

# Run the actual inference runs
inference_times = []
print("\nStarting measured iterations...")

nvtx.range_push("SDXL_Multiple_Inference")

for i in range(N_ITERATIONS):
    torch.cuda.synchronize()
    start_time = time.time()
    
    nvtx.range_push(f"Iteration_{i}")
    images = generate_images(pipe, prompts, device)
    nvtx.range_pop()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print(f"Iteration {i+1}/{N_ITERATIONS}: {inference_time:.3f} seconds")
    
    # Save batch images from last iteration
    if i == N_ITERATIONS - 1:
        if not is_distributed or (is_distributed and dist.get_rank() == 0):
            for j, image in enumerate(images):
                image.save(f"outputs/output_{j}.png")

nvtx.range_pop()

# Calculate and print statistics
avg_time = sum(inference_times) / len(inference_times)
min_time = min(inference_times)
max_time = max(inference_times)
print(f"\nResults over {N_ITERATIONS} iterations:")
print(f"Average inference time: {avg_time:.3f} seconds")
print(f"Min inference time: {min_time:.3f} seconds")
print(f"Max inference time: {max_time:.3f} seconds")
print(f"seconds per image: {avg_time/BATCH_SIZE:.3f} seconds/img")

# Cleanup
if is_distributed:
    dist.destroy_process_group()


    