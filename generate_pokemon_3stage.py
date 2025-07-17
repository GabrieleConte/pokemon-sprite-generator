#!/usr/bin/env python3
"""
Complete inference script for Pokemon sprite generation using the 3-stage trained model.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.vae_decoder import VAEEncoder, VAEDecoder
from src.models.text_encoder import TextEncoder
from src.models.unet import UNet
from src.training.vae_trainer import load_config
from src.utils import get_device


class PokemonGenerator:
    """
    Complete Pokemon sprite generator using VAE + U-Net + Text Encoder.
    """
    
    def __init__(self, vae_checkpoint_path: str, diffusion_checkpoint_path: str, 
                 config_path: str, device: Optional[str] = None):
        """
        Initialize the Pokemon generator.
        
        Args:
            vae_checkpoint_path: Path to VAE checkpoint
            diffusion_checkpoint_path: Path to diffusion checkpoint
            config_path: Path to training configuration
            device: Device to use ('cuda', 'mps', or 'cpu')
        """
        self.device = device or get_device()
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = load_config(config_path)
        model_config = self.config['model']
        
        # Initialize models
        self.text_encoder = TextEncoder(
            model_name=model_config['bert_model'],
            hidden_dim=model_config['text_embedding_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers']
        ).to(self.device)
        
        self.vae_encoder = VAEEncoder(
            input_channels=3,
            latent_dim=model_config.get('latent_dim', 512)
        ).to(self.device)
        
        self.vae_decoder = VAEDecoder(
            latent_dim=model_config.get('latent_dim', 512),
            text_dim=model_config['text_embedding_dim'],
            output_channels=3
        ).to(self.device)
        
        self.unet = UNet(
            latent_dim=model_config.get('latent_dim', 512),
            text_dim=model_config['text_embedding_dim'],
            time_emb_dim=model_config.get('time_emb_dim', 128),
            num_heads=model_config.get('num_heads', 8)
        ).to(self.device)
        
        # Load model weights
        self.load_checkpoints(vae_checkpoint_path, diffusion_checkpoint_path)
        
        # Set to evaluation mode
        self.text_encoder.eval()
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        self.unet.eval()
        
        # Noise scheduler parameters
        self.num_timesteps = model_config.get('num_timesteps', 1000)
        self.beta_start = model_config.get('beta_start', 0.0001)
        self.beta_end = model_config.get('beta_end', 0.02)
        
        # Initialize noise scheduler
        self.setup_noise_scheduler()
        
    def load_checkpoints(self, vae_checkpoint_path: str, diffusion_checkpoint_path: str):
        """Load model weights from checkpoints."""
        # Load VAE checkpoint
        print(f"Loading VAE checkpoint from {vae_checkpoint_path}")
        vae_checkpoint = torch.load(vae_checkpoint_path, map_location=self.device)
        
        self.vae_encoder.load_state_dict(vae_checkpoint['vae_encoder_state_dict'])
        self.vae_decoder.load_state_dict(vae_checkpoint['vae_decoder_state_dict'])
        
        # Load text encoder if available
        if 'text_encoder_state_dict' in vae_checkpoint:
            self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'])
        
        # Load diffusion checkpoint
        print(f"Loading diffusion checkpoint from {diffusion_checkpoint_path}")
        diffusion_checkpoint = torch.load(diffusion_checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(diffusion_checkpoint['unet_state_dict'])
        
        print("All model weights loaded successfully!")
        
    def setup_noise_scheduler(self):
        """Setup noise scheduler for diffusion sampling."""
        # Linear schedule for betas
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
    
    def ddpm_sample(self, text_emb: torch.Tensor, num_inference_steps: int = 50) -> torch.Tensor:
        """
        Sample from the diffusion model using DDPM.
        
        Args:
            text_emb: Text embeddings [batch_size, seq_len, text_dim]
            num_inference_steps: Number of denoising steps
            
        Returns:
            Denoised latent representation
        """
        batch_size = text_emb.shape[0]
        
        # Start from pure noise
        latent = torch.randn(
            batch_size, 
            self.config['model'].get('latent_dim', 512), 
            3, 3, 
            device=self.device
        )
        
        # Sampling timesteps
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device
        )
        
        # DDPM sampling loop
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full((batch_size,), t.item(), dtype=torch.long, device=self.device)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.unet(latent, t_tensor, text_emb)
            
            # Compute denoising step
            if i < len(timesteps) - 1:
                # Not the last step
                next_t = timesteps[i + 1]
                alpha_t = self.alphas[t]
                alpha_t_next = self.alphas[next_t]
                
                # Compute denoised latent
                latent = (latent - (1 - alpha_t) / torch.sqrt(1 - self.alphas_cumprod[t]) * predicted_noise) / torch.sqrt(alpha_t)
                
                # Add noise for next step
                if next_t > 0:
                    noise = torch.randn_like(latent)
                    latent = torch.sqrt(alpha_t_next) * latent + torch.sqrt(1 - alpha_t_next) * noise
            else:
                # Last step - final denoising
                latent = (latent - (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_cumprod[t]) * predicted_noise) / torch.sqrt(self.alphas[t])
        
        return latent
    
    def generate(self, descriptions: list, num_inference_steps: int = 50) -> list:
        """
        Generate Pokemon sprites from text descriptions.
        
        Args:
            descriptions: List of text descriptions
            num_inference_steps: Number of diffusion denoising steps
            
        Returns:
            List of generated images as PIL Images
        """
        with torch.no_grad():
            # Encode text
            text_emb = self.text_encoder(descriptions)
            
            # Sample from diffusion model
            latent = self.ddpm_sample(text_emb, num_inference_steps)
            
            # Decode to images
            generated_images = self.vae_decoder(latent, text_emb)
            
            # Convert to PIL Images
            images = []
            for img_tensor in generated_images:
                # Denormalize from [-1, 1] to [0, 1]
                img_tensor = (img_tensor + 1.0) / 2.0
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # Convert to PIL
                img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                images.append(img_pil)
            
            return images


def main():
    parser = argparse.ArgumentParser(description='Generate Pokemon sprites from text descriptions')
    parser.add_argument('--vae-checkpoint', 
                       required=True,
                       help='Path to VAE checkpoint')
    parser.add_argument('--diffusion-checkpoint', 
                       required=True,
                       help='Path to diffusion checkpoint')
    parser.add_argument('--config', 
                       default='config/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--description', 
                       required=True,
                       help='Text description of the Pokemon to generate')
    parser.add_argument('--output', 
                       default='generated_pokemon.png',
                       help='Output image path')
    parser.add_argument('--num-inference-steps', 
                       type=int,
                       default=50,
                       help='Number of diffusion inference steps')
    parser.add_argument('--device', 
                       choices=['cuda', 'mps', 'cpu'],
                       default=None,
                       help='Device to use for generation')
    
    args = parser.parse_args()
    
    # Check checkpoint files exist
    if not os.path.exists(args.vae_checkpoint):
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
    
    if not os.path.exists(args.diffusion_checkpoint):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {args.diffusion_checkpoint}")
    
    # Initialize generator
    print("Initializing Pokemon generator...")
    generator = PokemonGenerator(
        vae_checkpoint_path=args.vae_checkpoint,
        diffusion_checkpoint_path=args.diffusion_checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Generate image
    print(f"Generating Pokemon sprite for: '{args.description}'")
    images = generator.generate([args.description], num_inference_steps=args.num_inference_steps)
    
    # Save image
    images[0].save(args.output)
    print(f"Generated image saved to: {args.output}")


if __name__ == "__main__":
    main()
