


from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")





import time
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

# Configuration
N_WARMUP = 3  # Number of warmup iterations
N_ITERATIONS = 10  # Number of measured iterations
prompt = "A majestic mountain landscape at sunset"

# Enable CUDA profiling
torch.cuda.empty_cache()
# Optionally compile the model for potential speedup
if False:
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
    print("Model compilation enabled")

# Warmup runs
print("Performing warmup iterations...")
for i in range(N_WARMUP):
    image = pipe(prompt).images[0]
    #image.save(f"images/warmup_{i+1}.png")
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


    