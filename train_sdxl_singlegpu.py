import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from tqdm import tqdm
import bitsandbytes as bnb
from torch.cuda.amp import autocast


# Parse command line arguments
parser = argparse.ArgumentParser(description="Train Stable Diffusion XL on random noise")
parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size for gradient accumulation")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples in dataset")
parser.add_argument("--image_size", type=int, default=128, help="Image size for training")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")

args = parser.parse_args()

# Calculate gradient accumulation steps
gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)
    


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


def main():
    
    device = torch.device(f"cuda")

    # Model with FP16
    model = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
    ).unet.to(dtype=torch.float16, device=device)
    model.enable_gradient_checkpointing()


    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate)
    

    # Enable gradient checkpointing to reduce memory usage

    # Noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Dataset and dataloader
    dataset = RandomNoiseDataset(size=args.dataset_size, img_size=args.image_size)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.micro_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )    

    # Training parameters
    total_steps = args.steps  # Total number of training steps

    # Training loop
    model.train()
    total_loss = 0

    # Create progress bar
    
    progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)
    
    step = 0
    accumulated_samples = 0

    while step < total_steps:
        # Reset gradients at the beginning of each effective batch
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation loop
        accumulated_loss = 0
        
        for _ in range(gradient_accumulation_steps):
            # Get next batch
            try:
                batch = next(dataloader_iter)
            except (StopIteration, NameError):
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
                
            # Move all batch tensors to device
            latents = batch['latent'].to(dtype=torch.float16, device=device)
            text_embeddings = batch['text_embeddings'].to(dtype=torch.float16, device=device)
            text_embeds = batch['text_embeds'].to(dtype=torch.float16, device=device)
            time_ids = batch['time_ids'].to(dtype=torch.float16, device=device)
            micro_batch_size = latents.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, 
                (micro_batch_size,), 
            ).long().to(device)
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            # Forward pass with autocast
            with autocast():
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
            
            # Backward pass
            scaled_loss.backward()
            
                
            accumulated_loss += loss.item()
            accumulated_samples += micro_batch_size
            
            # If we've processed an entire effective batch, update weights
            if accumulated_samples >= args.batch_size or (step == total_steps - 1 and _ == gradient_accumulation_steps - 1):
                # Clip gradients to prevent explosion (common with FP16)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Update step counter and progress
                step += 1
                total_loss += accumulated_loss / gradient_accumulation_steps
                
                # Reset accumulation counters
                accumulated_samples = 0
                accumulated_loss = 0
                

                if step % 10 == 0:  # Update loss display every 10 steps
                    progress_bar.set_postfix({"loss": f"{total_loss/10:.4f}"})
                    total_loss = 0
                progress_bar.update(1)
                    
                if step >= total_steps:
                    break

    progress_bar.close()


if __name__ == "__main__":
    main()
