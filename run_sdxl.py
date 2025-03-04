from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch
import time
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import torch.distributed as dist

# Configuration
N_WARMUP = 3  # Number of warmup iterations
N_ITERATIONS = 10  # Number of measured iterations
prompt = "A majestic mountain landscape at sunset"

def setup_tensor_parallel():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def create_pipeline_with_tensor_parallel():
    print("Loading model with tensor parallelism...")
    
    # Initialize distributed setup
    setup_tensor_parallel()
    
    # Load base model
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
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
    
    # Move other components to current device
    device = torch.device(f"cuda:{dist.get_rank()}")
    pipe.text_encoder = pipe.text_encoder.to(device)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(device)
    pipe.vae = pipe.vae.to(device)
    
    print("\nModel distribution:")
    print(f"World size: {dist.get_world_size()}")
    print(f"Current rank: {dist.get_rank()}")
    print(f"Device: {device}")
    
    return pipe

# Initialize pipeline with tensor parallelism
pipe = create_pipeline_with_tensor_parallel()

# Enable CUDA profiling
torch.cuda.empty_cache()

# Warmup runs
print("\nPerforming warmup iterations...")
for i in range(N_WARMUP):
    with torch.inference_mode():
        nvtx.range_push(f"Warmup_{i}")
        image = pipe(prompt).images[0]
        nvtx.range_pop()
    print(f"Warmup iteration {i+1}/{N_WARMUP} complete")

# Profile the actual inference runs
inference_times = []
print("\nStarting measured iterations...")

nvtx.range_push("SDXL_Multiple_Inference")
profiler.start()

for i in range(N_ITERATIONS):
    torch.cuda.synchronize()
    start_time = time.time()
    
    nvtx.range_push(f"Iteration_{i}")
    with torch.inference_mode():
        image = pipe(prompt).images[0]
    nvtx.range_pop()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print(f"Iteration {i+1}/{N_ITERATIONS}: {inference_time:.3f} seconds")

profiler.stop()
nvtx.range_pop()

# Calculate and print statistics
avg_time = sum(inference_times) / len(inference_times)
min_time = min(inference_times)
max_time = max(inference_times)
print(f"\nResults over {N_ITERATIONS} iterations:")
print(f"Average inference time: {avg_time:.3f} seconds")
print(f"Min inference time: {min_time:.3f} seconds")
print(f"Max inference time: {max_time:.3f} seconds")

# Save the last generated image
if dist.get_rank() == 0:  # Only save on main process
    image.save("output.png")

# Cleanup
dist.destroy_process_group()


    