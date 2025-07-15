#!/usr/bin/env python3
"""
Test and generate Pokemon sprites from text descriptions using a trained model.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import load_config
from src.models.pokemon_generator import PokemonSpriteGenerator

def load_trained_model(checkpoint_path: str, config_path: str):
    """Load a trained model from checkpoint."""
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Initialize model
    model_config = config['model']
    model = PokemonSpriteGenerator(
        model_name=model_config['bert_model'],
        text_embedding_dim=model_config['text_embedding_dim'],
        noise_dim=model_config['noise_dim'],
        nhead=model_config['nhead'],
        num_encoder_layers=model_config['num_encoder_layers']
    )
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded successfully! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def tensor_to_image(tensor):
    """Convert a tensor to a PIL image."""
    # Move to CPU and convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Remove batch dimension if present
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    img_array = tensor.numpy().transpose(1, 2, 0)
    
    # Convert to PIL Image
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def generate_single_image(model, description, device='cpu', save_path=None):
    """Generate a single image from a description."""
    model.to(device)
    
    with torch.no_grad():
        # Generate image
        generated_tensor = model([description])
        
        # Convert to PIL Image
        image = tensor_to_image(generated_tensor)
        
        # Save if path provided
        if save_path:
            image.save(save_path)
            print(f"Image saved to: {save_path}")
        
        return image

def generate_variations(model, description, num_variations=4, device='cpu', save_dir=None):
    """Generate multiple variations of the same description."""
    model.to(device)
    
    images = []
    
    print(f"Generating {num_variations} variations for: '{description}'")
    
    with torch.no_grad():
        for i in range(num_variations):
            # Generate image (model uses different noise each time)
            generated_tensor = model([description])
            image = tensor_to_image(generated_tensor)
            images.append(image)
            
            # Save individual image
            if save_dir:
                save_path = Path(save_dir) / f"variation_{i+1}.png"
                image.save(save_path)
                print(f"Variation {i+1} saved to: {save_path}")
    
    # Create a grid of variations
    if save_dir:
        grid_path = Path(save_dir) / "variations_grid.png"
        create_image_grid(images, grid_path, description)
    
    return images

def generate_batch(model, descriptions, device='cpu', save_dir=None):
    """Generate images for multiple descriptions."""
    model.to(device)
    
    images = []
    
    print(f"Generating images for {len(descriptions)} descriptions...")
    
    with torch.no_grad():
        for i, description in enumerate(descriptions):
            generated_tensor = model([description])
            image = tensor_to_image(generated_tensor)
            images.append(image)
            
            # Save individual image
            if save_dir:
                save_path = Path(save_dir) / f"generated_{i+1:03d}.png"
                image.save(save_path)
                print(f"Generated image {i+1}: {description[:50]}...")
    
    # Create a grid of all images
    if save_dir:
        grid_path = Path(save_dir) / "batch_grid.png"
        create_image_grid(images, grid_path, f"Batch of {len(descriptions)} Pokemon")
    
    return images

def create_image_grid(images, save_path, title=None):
    """Create a grid of images and save it."""
    n_images = len(images)
    
    # Calculate grid size
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Handle single image case
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Image {i+1}", fontsize=12)
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grid saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate Pokemon sprites from text descriptions')
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', default='config/train_config.yaml', help='Path to config file')
    parser.add_argument('--description', type=str, help='Single description to generate')
    parser.add_argument('--descriptions-file', type=str, help='File with multiple descriptions (one per line)')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to generate')
    parser.add_argument('--output-dir', default='generated_samples', help='Output directory for generated images')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/mps/cpu/auto)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load trained model
    model = load_trained_model(args.checkpoint, args.config)
    
    # Default descriptions if none provided
    default_descriptions = [
        "A small yellow electric mouse Pokemon with red cheeks and a lightning bolt tail",
        "A large orange dragon Pokemon with blue wings and a flaming tail",
        "A blue turtle Pokemon with water cannons on its shell",
        "A green plant Pokemon with a large flower on its back",
        "A purple ghost Pokemon with glowing red eyes",
        "A brown ground Pokemon with sharp claws and armor",
        "A pink fairy Pokemon with large eyes and wings",
        "A silver steel Pokemon with metallic armor and spikes"
    ]
    
    # Generate images
    if args.description:
        # Single description
        if args.variations > 1:
            print(f"Generating {args.variations} variations...")
            generate_variations(model, args.description, args.variations, device, output_dir)
        else:
            print("Generating single image...")
            save_path = output_dir / "generated_single.png"
            generate_single_image(model, args.description, device, save_path)
    
    elif args.descriptions_file:
        # Multiple descriptions from file
        with open(args.descriptions_file, 'r') as f:
            descriptions = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(descriptions)} descriptions from file")
        generate_batch(model, descriptions, device, output_dir)
    
    else:
        # Use default descriptions
        print("No description provided, using default examples...")
        generate_batch(model, default_descriptions, device, output_dir)
    
    print(f"\nGeneration complete! Check the '{output_dir}' directory for results.")

if __name__ == "__main__":
    main()
