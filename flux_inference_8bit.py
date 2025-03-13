import torch
from diffusers import FluxTransformer2DModel, TorchAoConfig, FluxPipeline
import argparse
import time
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmark FLUX.1-schnell model generation speed')
parser.add_argument('--num_images', type=int, default=16, help='Total number of images to generate for benchmarking')
parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
parser.add_argument('--prompt', type=str, default="", help='Text prompt for generation')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
args = parser.parse_args()

model_id = "black-forest-labs/FLUX.1-schnell"
quantization_config = TorchAoConfig("int8wo")
torch_dtype = torch.bfloat16

# Load the pipeline
print("Loading model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    transformer=None,
    vae=None,
).to("cuda")

transformer = FluxTransformer2DModel.from_pretrained(
      model_id,
      subfolder="transformer",
      quantization_config=quantization_config,
      torch_dtype=torch.bfloat16,
)
pipe.transformer = transformer
pipe.to("cuda")
print("Model loaded successfully")

# Warmup run
print("Performing warmup run...")
_ = pipe(
    args.prompt, 
    num_inference_steps=8, 
    guidance_scale=0.0, 
    output_type="latent",
    num_images_per_prompt=args.batch_size,
    generator=torch.Generator("cuda").manual_seed(args.seed),
)
print("Warmup complete")

# Calculate number of runs needed to generate the target number of images
num_runs = (args.num_images + args.batch_size - 1) // args.batch_size  # Ceiling division
print(f"\nBenchmarking generation of {args.num_images} images with batch size {args.batch_size} ({num_runs} runs)...")

# Benchmark generation speed
run_times = []
total_images = 0

with tqdm(total=args.num_images, desc="Generating images", unit="img") as pbar:
    for run_idx in range(num_runs):
        
        # For the last batch, adjust batch size if needed
        current_batch_size = min(args.batch_size, args.num_images - total_images)
        
        # Time the generation process
        start_time = time.time()
        _ = pipe(
            args.prompt, 
            num_inference_steps=8, 
            guidance_scale=0.0, 
            output_type="latent",
            num_images_per_prompt=current_batch_size,
            generator=torch.Generator("cuda").manual_seed(0),
        )
        end_time = time.time()
        
        # Calculate and store the run time
        run_time = end_time - start_time
        run_times.append(run_time)
        
        # Update progress
        total_images += current_batch_size
        pbar.update(current_batch_size)
        
        # Display current speed
        current_ips = current_batch_size / run_time
        pbar.set_postfix({"img/s": f"{current_ips:.2f}"})

# Calculate and display statistics
avg_time_per_run = sum(run_times) / len(run_times)
avg_time_per_image = sum(run_times) / args.num_images
images_per_second = 1.0 / avg_time_per_image

print("\nBenchmark Results:")
print(f"Total images generated: {args.num_images}")
print(f"Average time per batch: {avg_time_per_run:.4f} seconds")
print(f"Average time per image: {avg_time_per_image:.4f} seconds")
print(f"Images per second: {images_per_second:.2f}")
print(f"Total memory used: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
