#!/usr/bin/env python3
"""
Batch generation script to create multiple Pokemon sprites.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from generate_pokemon_3stage import PokemonGenerator

def batch_generate():
    """Generate multiple Pokemon sprites with different descriptions."""
    
    # Configuration
    config_path = "config/train_config.yaml"
    vae_checkpoint = "experiments/pokemon_3stage_vae/checkpoints/vae_best_model.pth"
    diffusion_checkpoint = "experiments/pokemon_3stage_diffusion/checkpoints/diffusion_best_model.pth"
    
    # Test descriptions
    descriptions = [
        "A fire-type Pokemon with orange flames and wings",
        "A grass-type Pokemon with green vines and flowers",
        "A water-type Pokemon with blue scales and fins",
        "An electric-type Pokemon with yellow fur and lightning",
        "A rock-type Pokemon with gray stone armor",
        "A psychic-type Pokemon with purple aura and mystic powers"
    ]
    
    # Initialize generator
    print("Initializing Pokemon generator...")
    generator = PokemonGenerator(
        vae_checkpoint_path=vae_checkpoint,
        diffusion_checkpoint_path=diffusion_checkpoint,
        config_path=config_path,
        device="mps"
    )
    
    # Generate sprites
    for i, description in enumerate(descriptions):
        print(f"\n[{i+1}/{len(descriptions)}] Generating: {description}")
        
        output_path = f"generated_batch_{i+1:02d}.png"
        image = generator.generate(description, num_inference_steps=50)
        generator.save_image(image, output_path)
        
        print(f"Saved to: {output_path}")
    
    print("\n✅ Batch generation completed!")

if __name__ == "__main__":
    batch_generate()
