#!/usr/bin/env python3
"""
Interactive evaluation script for the Pokemon Sprite Generator.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import load_config
from src.models.pokemon_generator import PokemonSpriteGenerator
from PIL import Image
import numpy as np

def load_model(checkpoint_path, config_path='config/train_config.yaml'):
    """Load the trained model."""
    config = load_config(config_path)
    model_config = config['model']
    
    model = PokemonSpriteGenerator(
        model_name=model_config['bert_model'],
        text_embedding_dim=model_config['text_embedding_dim'],
        noise_dim=model_config['noise_dim'],
        nhead=model_config['nhead'],
        num_encoder_layers=model_config['num_encoder_layers']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    img_array = tensor.numpy().transpose(1, 2, 0)
    img_array = (img_array * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def interactive_session():
    """Run an interactive generation session."""
    print("🎮 Interactive Pokemon Sprite Generator")
    print("=" * 50)
    
    # Get checkpoint path
    checkpoint_path = input("Enter checkpoint path: ").strip()
    if not checkpoint_path:
        print("❌ Checkpoint path is required!")
        return
    
    # Load model
    try:
        print("Loading model...")
        model = load_model(checkpoint_path)
        print("✅ Model loaded successfully!")
        
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        model.to(device)
        print(f"Using device: {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create output directory
    output_dir = Path("interactive_outputs")
    output_dir.mkdir(exist_ok=True)
    
    generation_count = 0
    
    print(f"\n🎨 Ready to generate! Images will be saved to '{output_dir}'")
    print("Commands:")
    print("  - Enter any description to generate a Pokemon")
    print("  - 'random' for a random example")
    print("  - 'help' for more commands")
    print("  - 'quit' to exit")
    
    examples = [
        "A small yellow electric mouse Pokemon with red cheeks",
        "A large blue dragon Pokemon with crystal wings",
        "A green plant Pokemon with flower petals",
        "A fire Pokemon with orange flames and burning tail",
        "A ghost Pokemon with purple mist and glowing eyes",
    ]
    
    while True:
        try:
            print("\n" + "-" * 30)
            description = input("Enter description: ").strip()
            
            if description.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif description.lower() == 'help':
                print("Available commands:")
                print("  - Any text: Generate Pokemon from description")
                print("  - 'random': Generate from random example")
                print("  - 'examples': Show example descriptions")
                print("  - 'clear': Clear the screen")
                print("  - 'quit': Exit the program")
                continue
            
            elif description.lower() == 'random':
                import random
                description = random.choice(examples)
                print(f"Random description: {description}")
            
            elif description.lower() == 'examples':
                print("Example descriptions:")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
                continue
            
            elif description.lower() == 'clear':
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            elif not description:
                print("Please enter a description!")
                continue
            
            # Generate image
            generation_count += 1
            print(f"🎨 Generating Pokemon #{generation_count}...")
            
            with torch.no_grad():
                generated_tensor = model([description])
                image = tensor_to_image(generated_tensor)
            
            # Save image
            filename = f"pokemon_{generation_count:03d}.png"
            save_path = output_dir / filename
            image.save(save_path)
            
            print(f"✅ Generated! Saved as '{save_path}'")
            print(f"   Description: {description}")
            print(f"   Image size: {image.size}")
            
            # Ask if user wants to see more variations
            while True:
                another = input("Generate another variation? (y/n): ").strip().lower()
                if another in ['y', 'yes']:
                    generation_count += 1
                    print(f"🎨 Generating variation #{generation_count}...")
                    
                    with torch.no_grad():
                        generated_tensor = model([description])
                        image = tensor_to_image(generated_tensor)
                    
                    filename = f"pokemon_{generation_count:03d}_variation.png"
                    save_path = output_dir / filename
                    image.save(save_path)
                    
                    print(f"✅ Variation generated! Saved as '{save_path}'")
                elif another in ['n', 'no']:
                    break
                else:
                    print("Please enter 'y' or 'n'")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again...")

if __name__ == "__main__":
    interactive_session()
