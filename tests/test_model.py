#!/usr/bin/env python3
"""
Quick test script to verify your trained Pokemon model works correctly.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_model_loading():
    """Test if we can load the model architecture."""
    try:
        from src.models.pokemon_generator import PokemonSpriteGenerator
        from src.training.trainer import load_config
        
        # Load config
        config_path = Path(__file__).parent.parent / 'config' / 'train_config.yaml'
        config = load_config(str(config_path))
        model_config = config['model']
        
        # Create model
        model = PokemonSpriteGenerator(
            model_name=model_config['bert_model'],
            text_embedding_dim=model_config['text_embedding_dim'],
            noise_dim=model_config['noise_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers']
        )
        
        print("✅ Model architecture loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_model_inference():
    """Test if the model can generate images."""
    try:
        from src.models.pokemon_generator import PokemonSpriteGenerator
        from src.training.trainer import load_config
        
        # Load config
        config = load_config('config/train_config.yaml')
        model_config = config['model']
        
        # Create model
        model = PokemonSpriteGenerator(
            model_name=model_config['bert_model'],
            text_embedding_dim=model_config['text_embedding_dim'],
            noise_dim=model_config['noise_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers']
        )
        
        model.eval()
        
        # Test inference
        test_description = "A small yellow electric mouse Pokemon with red cheeks"
        
        with torch.no_grad():
            output = model([test_description])
        
        print("✅ Model inference works!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model inference: {e}")
        return False

def test_checkpoint_loading(checkpoint_path):
    """Test loading a trained checkpoint."""
    try:
        from src.models.pokemon_generator import PokemonSpriteGenerator
        from src.training.trainer import load_config
        
        # Load config
        config_path = Path(__file__).parent.parent / 'config' / 'train_config.yaml'
        config = load_config(str(config_path))
        model_config = config['model']
        
        # Create model
        model = PokemonSpriteGenerator(
            model_name=model_config['bert_model'],
            text_embedding_dim=model_config['text_embedding_dim'],
            noise_dim=model_config['noise_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers']
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try different possible keys
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        print("✅ Checkpoint loaded successfully!")
        
        # Test inference with loaded model
        test_description = "A small yellow electric mouse Pokemon with red cheeks"
        
        with torch.no_grad():
            output = model([test_description])
        
        print("✅ Trained model inference works!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def main():
    print("🧪 Testing Pokemon Sprite Generator Model")
    print("=" * 50)
    
    # Test 1: Model architecture
    print("\n1. Testing model architecture...")
    if not test_model_loading():
        return
    
    # Test 2: Model inference
    print("\n2. Testing model inference...")
    if not test_model_inference():
        return
    
    # Test 3: Checkpoint loading (if provided)
    import sys
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"\n3. Testing checkpoint loading...")
        if not test_checkpoint_loading(checkpoint_path):
            return
    else:
        print("\n3. Checkpoint loading test skipped (no checkpoint provided)")
        print("   Usage: python test_model.py <checkpoint_path>")
    
    print("\n🎉 All tests passed! Your model is ready to use.")
    print("\nNext steps:")
    print("1. Train your model: python train.py")
    print("2. Test generation: python test_generation.py --checkpoint <checkpoint_path>")
    print("3. Generate samples: python test_generation.py --checkpoint <checkpoint_path> --description 'Your Pokemon description'")

if __name__ == "__main__":
    main()
