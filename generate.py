#!/usr/bin/env python3
"""
Generation script for Pokemon Sprite Generator.

This script generates Pokemon sprites from text descriptions using
a trained model.
"""

import argparse
import os
import torch
from PIL import Image
import numpy as np

from src.models import PokemonSpriteGenerator
from src.utils import (
    load_config, get_device, tensor_to_image, save_image_grid,
    save_attention_visualization
)


def generate_sprites(model, texts, device, num_samples=1, temperature=1.0, 
                    output_dir='outputs', save_attention=False):
    """Generate Pokemon sprites from text descriptions."""
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f"Generating sprite for: '{text}'")
            
            # Generate images
            if save_attention:
                generated_images, attention_info = model.generate(
                    [text], num_samples=num_samples, temperature=temperature,
                    return_attention=True
                )
                attention_weights = attention_info['attention_weights']
            else:
                generated_images = model.generate(
                    [text], num_samples=num_samples, temperature=temperature,
                    return_attention=False
                )
            
            # Save generated images
            for j in range(num_samples):
                image = tensor_to_image(generated_images[j])
                
                # Create filename
                safe_text = text.replace(' ', '_').replace(',', '').replace('.', '')[:50]
                filename = f"generated_{i:03d}_{safe_text}_sample_{j:02d}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                image.save(filepath)
                print(f"Saved: {filepath}")
                
                # Save attention visualization if requested
                if save_attention and j == 0:  # Only for first sample
                    # Get tokens for attention visualization
                    input_ids, _ = model.text_encoder.tokenize([text])
                    tokens = model.text_encoder.tokenizer.convert_ids_to_tokens(input_ids[0])
                    
                    # Remove special tokens
                    tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
                    
                    # Save attention visualization
                    attention_filename = f"attention_{i:03d}_{safe_text}.png"
                    attention_filepath = os.path.join(output_dir, attention_filename)
                    
                    save_attention_visualization(
                        attention_weights[0], tokens, generated_images[0], attention_filepath
                    )
                    print(f"Saved attention map: {attention_filepath}")
            
            # Save grid of all samples for this text
            if num_samples > 1:
                grid_filename = f"grid_{i:03d}_{safe_text}.png"
                grid_filepath = os.path.join(output_dir, grid_filename)
                save_image_grid(generated_images, grid_filepath, nrow=4)
                print(f"Saved grid: {grid_filepath}")


def interactive_generation(model, device, output_dir='outputs'):
    """Interactive generation mode."""
    print("Interactive Pokemon Sprite Generation")
    print("Enter text descriptions (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        text = input("Enter description: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            print("Please enter a valid description.")
            continue
        
        # Get generation parameters
        try:
            num_samples = int(input("Number of samples (default 1): ") or "1")
            temperature = float(input("Temperature (default 1.0): ") or "1.0")
            save_attention = input("Save attention maps? (y/n, default n): ").lower().startswith('y')
        except ValueError:
            print("Invalid input, using defaults.")
            num_samples = 1
            temperature = 1.0
            save_attention = False
        
        # Generate sprites
        generate_sprites(
            model, [text], device, num_samples, temperature, 
            output_dir, save_attention
        )
        
        print(f"Generated {num_samples} sprite(s) for: '{text}'")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='Generate Pokemon sprites from text')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--text', type=str, nargs='+', default=None,
                       help='Text descriptions to generate sprites for')
    parser.add_argument('--file', type=str, default=None,
                       help='File containing text descriptions (one per line)')
    parser.add_argument('--output', type=str, default='outputs/generated',
                       help='Output directory for generated images')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate per text')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--save-attention', action='store_true',
                       help='Save attention weight visualizations')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive generation mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = PokemonSpriteGenerator(
        vocab_size=config['model']['text_encoder']['vocab_size'],
        embedding_dim=config['model']['text_encoder']['embedding_dim'],
        num_heads=config['model']['text_encoder']['num_heads'],
        num_layers=config['model']['text_encoder']['num_layers'],
        hidden_dim=config['model']['text_encoder']['hidden_dim'],
        max_seq_length=config['model']['text_encoder']['max_seq_length'],
        latent_dim=config['model']['decoder']['latent_dim'],
        noise_dim=config['model']['decoder']['noise_dim'],
        image_size=config['model']['decoder']['image_size'],
        num_channels=config['model']['decoder']['num_channels'],
        base_channels=config['model']['decoder']['base_channels'],
        attention_dim=config['model']['attention']['hidden_dim'],
        dropout=config['model']['text_encoder']['dropout']
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded successfully. Parameters: {model.count_parameters():,}")
    
    # Determine texts to generate
    texts = []
    
    if args.text:
        texts.extend(args.text)
    
    if args.file:
        with open(args.file, 'r') as f:
            file_texts = [line.strip() for line in f if line.strip()]
        texts.extend(file_texts)
    
    # Interactive mode
    if args.interactive:
        interactive_generation(model, device, args.output)
        return
    
    # Default texts if none provided
    if not texts:
        texts = [
            "A small yellow electric mouse Pokemon with red cheeks and a lightning bolt tail",
            "A large orange dragon Pokemon with blue wings and a flaming tail", 
            "A blue turtle Pokemon with water cannons on its shell",
            "A green plant Pokemon with a large flower on its back and toxic spores",
            "A purple psychic Pokemon with a spoon in each hand and mystical powers",
            "A red and black fire dog Pokemon with flame patterns on its body",
            "An ice-type Pokemon with crystalline blue body and freezing breath",
            "A rock-type Pokemon with a hard gray body and boulder-like appearance"
        ]
        print("No texts provided, using default descriptions.")
    
    print(f"Generating sprites for {len(texts)} descriptions...")
    
    # Generate sprites
    generate_sprites(
        model, texts, device, args.num_samples, args.temperature,
        args.output, args.save_attention
    )
    
    print(f"Generation complete! Images saved to: {args.output}")


if __name__ == '__main__':
    main()