#!/usr/bin/env python3
"""
Inference script for generating Pokemon sprites from text descriptions.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.final_trainer import FinalPokemonGenerator, load_config
from src.utils import get_device


def denormalize_image(image_tensor):
    """Convert tensor from [-1, 1] to [0, 1] range."""
    return (image_tensor + 1.0) / 2.0


def tensor_to_pil(image_tensor):
    """Convert tensor to PIL Image."""
    # Denormalize and convert to numpy
    image = denormalize_image(image_tensor)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def generate_pokemon(model, descriptions, device, num_samples=1):
    """
    Generate Pokemon images from text descriptions.
    
    Args:
        model: Trained Pokemon generator model
        descriptions: List of text descriptions
        device: Device to run inference on
        num_samples: Number of samples to generate per description
        
    Returns:
        List of generated images (PIL Images)
    """
    model.eval()
    generated_images = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate images
            output = model(descriptions)
            
            # Convert each image to PIL
            for i in range(output.size(0)):
                pil_image = tensor_to_pil(output[i])
                generated_images.append(pil_image)
    
    return generated_images


def main():
    parser = argparse.ArgumentParser(description='Generate Pokemon sprites from text')
    parser.add_argument('--checkpoint', 
                       required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', 
                       default='config/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--descriptions', 
                       nargs='+',
                       default=["A small electric mouse Pokemon with yellow fur and red cheeks"],
                       help='Text descriptions to generate Pokemon from')
    parser.add_argument('--num-samples', 
                       type=int,
                       default=1,
                       help='Number of samples to generate per description')
    parser.add_argument('--output-dir', 
                       default='generated_pokemon',
                       help='Directory to save generated images')
    parser.add_argument('--save-grid', 
                       action='store_true',
                       help='Save a grid of all generated images')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model = FinalPokemonGenerator(
        vae_path=args.checkpoint,  # This will be handled differently in practice
        text_encoder_config=config['model']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully!")
    print(f"Will generate {args.num_samples} sample(s) for each description")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate images
    print("Generating Pokemon...")
    generated_images = generate_pokemon(
        model, 
        args.descriptions, 
        device, 
        num_samples=args.num_samples
    )
    
    # Save individual images
    image_idx = 0
    for desc_idx, description in enumerate(args.descriptions):
        print(f"Description {desc_idx + 1}: {description}")
        
        for sample_idx in range(args.num_samples):
            image = generated_images[image_idx]
            
            # Create filename
            filename = f"pokemon_{desc_idx+1}_sample_{sample_idx+1}.png"
            filepath = output_dir / filename
            
            # Save image
            image.save(filepath)
            print(f"  Saved: {filepath}")
            
            image_idx += 1
    
    # Save grid if requested
    if args.save_grid:
        print("Creating grid of all generated images...")
        
        # Calculate grid size
        total_images = len(generated_images)
        cols = min(4, total_images)
        rows = (total_images + cols - 1) // cols
        
        # Create grid
        if total_images == 1:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.imshow(generated_images[0])
            ax.axis('off')
            ax.set_title("Generated Pokemon", fontsize=8)
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            
            # Handle different subplot configurations
            if rows == 1:
                axes = [axes] if cols == 1 else list(axes)
            
            for i, image in enumerate(generated_images):
                row = i // cols
                col = i % cols
                
                if rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row][col]
                
                ax.imshow(image)
                ax.axis('off')
                
                # Add description as title
                desc_idx = i // args.num_samples
                sample_idx = i % args.num_samples
                title = f"Desc {desc_idx+1}, Sample {sample_idx+1}"
                ax.set_title(title, fontsize=8)
            
            # Hide unused subplots
            for i in range(total_images, rows * cols):
                row = i // cols
                col = i % cols
                
                if rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row][col]
                ax.axis('off')
        
        plt.tight_layout()
        grid_path = output_dir / "generated_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Grid saved: {grid_path}")
    
    print(f"\nGeneration complete! Check {output_dir} for results.")


if __name__ == "__main__":
    main()
