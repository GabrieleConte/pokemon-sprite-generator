#!/usr/bin/env python3
"""
Test script to verify the Pokemon Sprite Generator implementation.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import PokemonSpriteGenerator
from data import create_data_loaders
from utils import get_device, set_seed


def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing model creation...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = PokemonSpriteGenerator(
        vocab_size=30522,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=512,
        max_seq_length=128,
        latent_dim=256,
        noise_dim=100,
        image_size=215,
        num_channels=3,
        base_channels=64,
        attention_dim=256,
        dropout=0.1
    )
    
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test tokenization
    test_texts = [
        "A small yellow electric mouse Pokemon with red cheeks",
        "A large orange dragon Pokemon with blue wings"
    ]
    
    input_ids, attention_mask = model.text_encoder.tokenize(test_texts)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Test forward pass
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        generated_images = model(input_ids, attention_mask)
        print(f"Generated images shape: {generated_images.shape}")
    
    # Test generation
    with torch.no_grad():
        generated_images = model.generate(test_texts, num_samples=2)
        print(f"Generated images shape (2 samples): {generated_images.shape}")
    
    print("Model creation test passed!")
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='data',
        batch_size=2,
        num_workers=0  # Use 0 for testing
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch descriptions: {len(batch['descriptions'])}")
        print(f"Sample description: {batch['descriptions'][0][:100]}...")
        break
    
    print("Data loading test passed!")
    return True


def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")
    
    device = get_device()
    model = PokemonSpriteGenerator().to(device)
    
    # Create dummy data
    batch_size = 2
    generated_images = torch.randn(batch_size, 3, 215, 215).to(device)
    target_images = torch.randn(batch_size, 3, 215, 215).to(device)
    attention_weights = torch.softmax(torch.randn(batch_size, 128), dim=1).to(device)
    
    # Compute loss
    total_loss, loss_dict = model.compute_loss(
        generated_images, target_images, attention_weights
    )
    
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Reconstruction loss: {loss_dict['reconstruction_loss'].item():.6f}")
    print(f"Attention loss: {loss_dict['attention_loss'].item():.6f}")
    
    print("Loss computation test passed!")
    return True


def main():
    """Run all tests."""
    print("Running Pokemon Sprite Generator tests...")
    print("=" * 50)
    
    try:
        test_model_creation()
        test_data_loading()
        test_loss_computation()
        
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)