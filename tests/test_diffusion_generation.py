#!/usr/bin/env python3
"""
Test script to verify that the final trainer uses the diffusion process correctly.
"""

import torch
import yaml
from src.training.final_trainer import FinalPokemonGenerator
from src.utils import get_device

def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_diffusion_generation():
    """Test that the final generator uses diffusion process."""
    
    # Load configuration
    config = load_config("config/train_config.yaml")
    
    # Paths to checkpoints
    vae_checkpoint_path = "experiments/pokemon_3stage_vae/checkpoints/vae_best_model.pth"
    diffusion_checkpoint_path = "experiments/pokemon_3stage_diffusion/checkpoints/diffusion_best_model.pth"
    
    # Create generator
    device = get_device()
    generator = FinalPokemonGenerator(
        vae_path=vae_checkpoint_path,
        diffusion_path=diffusion_checkpoint_path,
        text_encoder_config=config['model']
    ).to(device)
    
    # Test with sample descriptions
    test_descriptions = [
        "A fire-type Pokemon with orange fur and flame tail",
        "A water-type Pokemon with blue scales and fins",
        "A grass-type Pokemon with green leaves and vines"
    ]
    
    print("Testing diffusion generation...")
    
    # Generate with different inference steps to verify diffusion is working
    for num_steps in [10, 25, 50]:
        print(f"\nGenerating with {num_steps} inference steps...")
        
        with torch.no_grad():
            generated_images = generator(test_descriptions, num_inference_steps=num_steps)
        
        print(f"Generated images shape: {generated_images.shape}")
        print(f"Generated images min: {generated_images.min():.4f}, max: {generated_images.max():.4f}")
        
        # Check that images are different with different inference steps
        # (This helps verify diffusion is actually running)
        if num_steps == 10:
            images_10_steps = generated_images.clone()
        elif num_steps == 50:
            images_50_steps = generated_images.clone()
            
    # Compare images from different inference steps
    diff = torch.abs(images_10_steps - images_50_steps).mean()
    print(f"\nDifference between 10-step and 50-step generation: {diff:.6f}")
    
    if diff > 0.001:
        print("✅ Diffusion process is working - images differ with different inference steps")
    else:
        print("⚠️  Images are too similar - diffusion process may not be working correctly")
    
    # Test model components
    print(f"\nModel components:")
    print(f"- VAE Encoder parameters: {sum(p.numel() for p in generator.vae_encoder.parameters())}")
    print(f"- VAE Decoder parameters: {sum(p.numel() for p in generator.vae_decoder.parameters())}")
    print(f"- U-Net parameters: {sum(p.numel() for p in generator.unet.parameters())}")
    print(f"- Text Encoder parameters: {sum(p.numel() for p in generator.text_encoder.parameters())}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}")

if __name__ == "__main__":
    test_diffusion_generation()
