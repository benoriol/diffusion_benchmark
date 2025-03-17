import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from tqdm import tqdm
import bitsandbytes as bnb
from torch.cuda.amp import autocast
import time
import numpy as np


from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
# Import FSDP modules
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from diffusers.models.attention import Attention

# Import NVTX for profiling
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    print("NVTX not available. Install with: pip install nvtx")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train Stable Diffusion XL on random noise")
parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size for gradient accumulation")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples in dataset")
parser.add_argument("--image_size", type=int, default=128, help="Image size for training")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
parser.add_argument("--profile", action="store_true", help="Enable NVTX profiling markers")
# FSDP specific arguments
parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD", 
                    choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                    help="FSDP sharding strategy")
parser.add_argument("--min_params_per_shard", type=int, default=1e6, 
                    help="Minimum parameters per shard for auto wrap policy")
parser.add_argument("--cpu_offload", action="store_true", 
                    help="Enable CPU offloading for FSDP")
parser.add_argument("--backward_prefetch", type=str, default="BACKWARD_PRE", 
                    choices=["BACKWARD_PRE", "BACKWARD_POST", "NO_PREFETCH"],
                    help="FSDP backward prefetch strategy")

args = parser.parse_args()

# Calculate gradient accumulation steps
gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)

# Profiling helper functions
def nvtx_range_push(msg):
    if NVTX_AVAILABLE and args.profile:
        nvtx.range_push(msg)

def nvtx_range_pop():
    if NVTX_AVAILABLE and args.profile:
        nvtx.range_pop()


def fsdp_setup(rank, world_size):
    nvtx_range_push("fsdp_setup")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", world_size=world_size, rank=rank)
    nvtx_range_pop()
    


# Dummy dataset
class RandomNoiseDataset(Dataset):
    def __init__(self, size=1000, img_size=128):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random latent image
        latent = torch.randn(4, self.img_size, self.img_size)
        
        # Random text embeddings (simulating CLIP text encoder output)
        # SDXL uses 77 tokens with 2048 dim hidden states from OpenCLIP ViT-bigG
        text_embeddings = torch.randn(77, 2048)
        
        # Random text embeds (simulating CLIP text encoder output)
        # SDXL uses 1280-dim text embeds
        text_embeds = torch.randn(1280)
        
        # Random time ids (simulating SDXL's added time embeddings)
        # SDXL uses 6-dim time embeddings
        time_ids = torch.randn(6)
        
        return {
            "latent": latent,
            "text_embeddings": text_embeddings,
            "text_embeds": text_embeds,
            "time_ids": time_ids
        }


def main(rank, world_size):
    nvtx_range_push("main")
    fsdp_setup(rank, world_size)
    # Device
    device = torch.device(f"cuda:{rank}")

    # Model with FP16
    nvtx_range_push("model_loading")
    model = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
    ).unet.to(dtype=torch.float16, device=device)
    model.enable_gradient_checkpointing()
    nvtx_range_pop()

    nvtx_range_push("fsdp_setup")
    # Configure FSDP mixed precision policy
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    # Configure CPU offloading if enabled
    cpu_offload = CPUOffload(offload_params=args.cpu_offload)
    
    # Configure backward prefetch
    if args.backward_prefetch == "BACKWARD_PRE":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif args.backward_prefetch == "BACKWARD_POST":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    else:
        backward_prefetch = None
    
    # Configure sharding strategy
    if args.sharding_strategy == "FULL_SHARD":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args.sharding_strategy == "SHARD_GRAD_OP":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding_strategy = ShardingStrategy.NO_SHARD
    
    # Define auto wrap policy
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=args.min_params_per_shard,
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        device_id=device,
    )
    
    # Create optimizer after FSDP wrapping
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate)
    nvtx_range_pop()

    # Noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Dataset and dataloader
    nvtx_range_push("dataset_setup")
    dataset = RandomNoiseDataset(size=args.dataset_size, img_size=args.image_size)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.micro_batch_size, 
        shuffle=False, 
        sampler=DistributedSampler(dataset, shuffle=True),
        num_workers=args.num_workers, 
        pin_memory=True
    )
    nvtx_range_pop()

    # Training parameters
    total_steps = args.steps  # Total number of training steps

    # Timing measurements
    batch_times = []
    
    # Training loop
    nvtx_range_push("training_loop")
    model.train()
    total_loss = 0

    # Create progress bar
    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)
    else:
        progress_bar = None
    step = 0
    accumulated_samples = 0

    while step < total_steps:
        nvtx_range_push(f"training_step_{step}")
        # Reset gradients at the beginning of each effective batch
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation loop
        accumulated_loss = 0
        
        # Start timing for this batch
        batch_start_time = time.time()
        
        for acc_step in range(gradient_accumulation_steps):
            nvtx_range_push(f"grad_accum_step_{acc_step}")
            # Get next batch
            nvtx_range_push("data_loading")
            try:
                batch = next(dataloader_iter)
            except (StopIteration, NameError):
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            nvtx_range_pop()
                
            # Move all batch tensors to device
            nvtx_range_push("data_to_device")
            latents = batch['latent'].to(dtype=torch.float16, device=device)
            text_embeddings = batch['text_embeddings'].to(dtype=torch.float16, device=device)
            text_embeds = batch['text_embeds'].to(dtype=torch.float16, device=device)
            time_ids = batch['time_ids'].to(dtype=torch.float16, device=device)
            micro_batch_size = latents.shape[0]
            nvtx_range_pop()
            
            # Sample timesteps
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, 
                (micro_batch_size,), 
            ).long().to(device)
            
            # Add noise to latents
            nvtx_range_push("add_noise")
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            nvtx_range_pop()
            
            # Forward pass with autocast
            nvtx_range_push("forward_pass")
            # Note: FSDP already handles mixed precision, so we can remove autocast
            # Pass all required arguments to the UNet
            noise_pred = model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs={
                    "text_embeds": text_embeds,
                    "time_ids": time_ids
                }
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Scale the loss to account for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps
            nvtx_range_pop()
            
            # Backward pass
            nvtx_range_push("backward_pass")
            scaled_loss.backward()
            nvtx_range_pop()
            
                
            accumulated_loss += loss.item()
            accumulated_samples += micro_batch_size
            
            # If we've processed an entire effective batch, update weights
            if accumulated_samples >= args.batch_size or (step == total_steps - 1 and acc_step == gradient_accumulation_steps - 1):
                # Clip gradients to prevent explosion (common with FP16)
                nvtx_range_push("grad_clip_and_step")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                nvtx_range_pop()
                
                # End timing for this batch
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                
                # Update step counter and progress
                step += 1
                total_loss += accumulated_loss / gradient_accumulation_steps
                
                # Reset accumulation counters
                accumulated_samples = 0
                accumulated_loss = 0
                

                if step % 10 == 0:  # Update loss display every 10 steps
                    if progress_bar is not None:
                        progress_bar.set_postfix({"loss": f"{total_loss/10:.4f}", "time/batch": f"{batch_time:.2f}s"})
                        total_loss = 0
                
                if progress_bar is not None:
                    progress_bar.update(1)
                    
                if step >= total_steps:
                    break
                
                # Start timing for next batch
                batch_start_time = time.time()
            
            nvtx_range_pop()  # End of grad_accum_step
        
        nvtx_range_pop()  # End of training_step

    if progress_bar is not None:
        progress_bar.close()
    
    # Print timing statistics
    if rank == 0 and batch_times:
        # Skip the first batch as it often includes compilation/warmup overhead
        if len(batch_times) > 1:
            batch_times = batch_times[1:]
        
        avg_time = np.mean(batch_times)
        median_time = np.median(batch_times)
        min_time = np.min(batch_times)
        max_time = np.max(batch_times)
        std_time = np.std(batch_times)
        
        print("\n===== Batch Timing Statistics =====")
        print(f"Average time per batch: {avg_time:.4f}s")
        print(f"Median time per batch: {median_time:.4f}s")
        print(f"Min time per batch: {min_time:.4f}s")
        print(f"Max time per batch: {max_time:.4f}s")
        print(f"Std dev of batch times: {std_time:.4f}s")
        print(f"Throughput: {args.batch_size / avg_time:.2f} samples/second")
        print(f"Total batches measured: {len(batch_times)}")
        print("==================================")
    
    nvtx_range_pop()  # End of training_loop
    
    destroy_process_group()
    nvtx_range_pop()  # End of main

if __name__ == "__main__":
    nvtx_range_push("program_start")
    world_size = torch.cuda.device_count()
    #world_size = 1
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    nvtx_range_pop()  # End of program
