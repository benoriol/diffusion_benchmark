from diffusers import DiffusionPipeline
import torch
import time
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
from accelerate import dispatch_model
from accelerate.utils import get_balanced_memory

# Configuration
N_WARMUP = 3  # Number of warmup iterations
N_ITERATIONS = 10  # Number of measured iterations
prompt = "A majestic mountain landscape at sunset"

def create_pipeline_with_tensor_parallel():
    print("Loading model with tensor parallelism...")
    
    # Get the maximum memory available on each GPU
    n_gpus = torch.cuda.device_count()
    max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024*1024*1024)-2}GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "16GiB"  # Reserve some CPU memory as buffer
    
    # Initialize pipeline with balanced device map for tensor parallelism
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        device_map="balanced",
        max_memory=max_memory
    )
    
    # Get balanced memory for each GPU
    max_memory_per_gpu = get_balanced_memory(
        pipe.unet,
        dtype=torch.float16,
        low_zero=False,
        no_split_module_classes=["CrossAttention", "Attention"]
    )
    
    print("\nModel distribution across devices:")
    print(f"UNet device map: {pipe.unet.hf_device_map}")
    print(f"Text Encoder 1: {pipe.text_encoder.device}")
    print(f"Text Encoder 2: {pipe.text_encoder_2.device}")
    print(f"VAE: {pipe.vae.device}")
    print(f"Memory per GPU: {max_memory_per_gpu}")
    
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
image.save("output.png")


    