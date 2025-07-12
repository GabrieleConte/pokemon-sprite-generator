#!/usr/bin/env python3
"""
Example usage of the Pokemon Sprite Generator.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import PokemonSpriteGenerator
from utils import get_device, set_seed, tensor_to_image


def main():
    """Example of using the Pokemon Sprite Generator."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = PokemonSpriteGenerator(
    )
    
    model = model.to(device)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Example text descriptions
    descriptions = [
        "A small yellow electric mouse Pokemon with red cheeks and a lightning bolt tail",
        "A large orange dragon Pokemon with blue wings and a flaming tail",
        "A blue turtle Pokemon with water cannons on its shell"
    ]
    
    # Generate sprites
    print("\nGenerating Pokemon sprites...")
    with torch.no_grad():
        # Generate single samples
        generated_images = model.generate(descriptions, num_samples=1)
        print(f"Generated {len(generated_images)} sprites with shape: {generated_images.shape}")
        
        # Generate multiple samples with different temperatures
        for temperature in [0.5, 1.0, 1.5]:
            print(f"\nGenerating with temperature {temperature}...")
            generated_images = model.generate(descriptions[:1], num_samples=3, temperature=temperature)
            print(f"Generated {len(generated_images)} sprites")
    
    # Example with attention visualization
    print("\nGenerating with attention visualization...")
    with torch.no_grad():
        generated_images, attention_info = model.generate(
            descriptions[:1], 
            num_samples=1, 
            return_attention=True
        )
        
        attention_weights = attention_info['attention_weights']
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Sample attention weights: {attention_weights[0][:10].tolist()}")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()