#!/usr/bin/env python3
"""
Comprehensive test suite for the Pokemon Sprite Generator.
"""

import torch
import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import PokemonSpriteGenerator, TextEncoder, CNNDecoder, AttentionMechanism
from data import create_data_loaders
from utils import get_device, set_seed, load_config, save_config


def test_text_encoder():
    """Test text encoder functionality."""
    print("Testing Text Encoder...")
    
    encoder = TextEncoder(
        vocab_size=30522,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=512,
        max_seq_length=128
    )
    
    # Test tokenization
    texts = ["A small yellow electric Pokemon", "A large dragon with wings"]
    input_ids, attention_mask = encoder.tokenize(texts)
    
    assert input_ids.shape == (2, 128), f"Expected (2, 128), got {input_ids.shape}"
    assert attention_mask.shape == (2, 128), f"Expected (2, 128), got {attention_mask.shape}"
    
    # Test forward pass
    encoder_outputs, pooled_output = encoder(input_ids, attention_mask)
    
    assert encoder_outputs.shape == (2, 128, 256), f"Expected (2, 128, 256), got {encoder_outputs.shape}"
    assert pooled_output.shape == (2, 256), f"Expected (2, 256), got {pooled_output.shape}"
    
    print("✓ Text Encoder test passed")


def test_cnn_decoder():
    """Test CNN decoder functionality."""
    print("Testing CNN Decoder...")
    
    decoder = CNNDecoder(
        latent_dim=256,
        noise_dim=100,
        image_size=215,
        num_channels=3,
        base_channels=64
    )
    
    # Test forward pass
    batch_size = 2
    text_features = torch.randn(batch_size, 256)
    noise = torch.randn(batch_size, 100)
    
    generated_images = decoder(text_features, noise)
    
    assert generated_images.shape == (2, 3, 215, 215), f"Expected (2, 3, 215, 215), got {generated_images.shape}"
    assert generated_images.min() >= -1.0 and generated_images.max() <= 1.0, "Images should be in [-1, 1] range"
    
    # Test multiple samples
    samples = decoder.generate_samples(text_features, num_samples=3)
    assert samples.shape == (2, 3, 3, 215, 215), f"Expected (2, 3, 3, 215, 215), got {samples.shape}"
    
    print("✓ CNN Decoder test passed")


def test_attention_mechanism():
    """Test attention mechanism functionality."""
    print("Testing Attention Mechanism...")
    
    attention = AttentionMechanism(
        encoder_dim=256,
        decoder_dim=256,
        hidden_dim=256
    )
    
    # Set to eval mode to disable dropout
    attention.eval()
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    encoder_outputs = torch.randn(batch_size, seq_len, 256)
    decoder_state = torch.randn(batch_size, 256)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    context_vector, attention_weights = attention(encoder_outputs, decoder_state, attention_mask)
    
    assert context_vector.shape == (2, 256), f"Expected (2, 256), got {context_vector.shape}"
    assert attention_weights.shape == (2, 128), f"Expected (2, 128), got {attention_weights.shape}"
    
    # Check attention weights sum to 1
    attention_sums = attention_weights.sum(dim=1)
    print(f"Attention weights sum: {attention_sums}")
    assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-3), f"Attention weights should sum to 1, got {attention_sums}"
    
    print("✓ Attention Mechanism test passed")


def test_complete_model():
    """Test complete model functionality."""
    print("Testing Complete Model...")
    
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
    
    # Test model creation
    param_count = model.count_parameters()
    assert param_count > 0, "Model should have trainable parameters"
    
    # Test forward pass
    texts = ["A small yellow electric Pokemon", "A large dragon with wings"]
    input_ids, attention_mask = model.text_encoder.tokenize(texts)
    
    generated_images = model(input_ids, attention_mask)
    assert generated_images.shape == (2, 3, 215, 215), f"Expected (2, 3, 215, 215), got {generated_images.shape}"
    
    # Test generation
    generated_images = model.generate(texts, num_samples=2)
    assert generated_images.shape == (4, 3, 215, 215), f"Expected (4, 3, 215, 215), got {generated_images.shape}"
    
    # Test with attention
    generated_images, attention_info = model.generate(texts, num_samples=1, return_attention=True)
    assert 'attention_weights' in attention_info, "Should return attention weights"
    assert 'cross_attention_weights' in attention_info, "Should return cross attention weights"
    
    # Test loss computation
    target_images = torch.randn(2, 3, 215, 215)
    attention_weights = torch.softmax(torch.randn(2, 128), dim=1)
    
    total_loss, loss_dict = model.compute_loss(generated_images, target_images, attention_weights)
    
    assert 'total_loss' in loss_dict, "Should have total loss"
    assert 'reconstruction_loss' in loss_dict, "Should have reconstruction loss"
    assert 'attention_loss' in loss_dict, "Should have attention loss"
    
    print("✓ Complete Model test passed")


def test_model_save_load():
    """Test model save and load functionality."""
    print("Testing Model Save/Load...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.pth")
        
        # Create and save model with matching configurations
        model1 = PokemonSpriteGenerator(
            embedding_dim=128, 
            hidden_dim=256, 
            latent_dim=128, 
            attention_dim=128
        )
        model1.save_model(model_path)
        
        # Load model with same configuration
        model2 = PokemonSpriteGenerator(
            embedding_dim=128, 
            hidden_dim=256, 
            latent_dim=128, 
            attention_dim=128
        )
        model2.load_model(model_path)
        
        # Test that models have same number of parameters
        assert model1.count_parameters() == model2.count_parameters(), "Models should have same number of parameters"
    
    print("✓ Model Save/Load test passed")


def test_configuration():
    """Test configuration functionality."""
    print("Testing Configuration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        
        # Test config
        config = {
            'model': {
                'text_encoder': {
                    'embedding_dim': 256,
                    'num_heads': 8
                },
                'decoder': {
                    'latent_dim': 256,
                    'image_size': 215
                }
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001
            }
        }
        
        # Save and load config
        save_config(config, config_path)
        loaded_config = load_config(config_path)
        
        assert loaded_config['model']['text_encoder']['embedding_dim'] == 256, "Config should be loaded correctly"
        assert loaded_config['training']['batch_size'] == 16, "Config should be loaded correctly"
    
    print("✓ Configuration test passed")


def test_data_loading():
    """Test data loading functionality."""
    print("Testing Data Loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=temp_dir,
            batch_size=2,
            num_workers=0
        )
        
        # Test data loading
        train_batch = next(iter(train_loader))
        assert 'images' in train_batch, "Batch should contain images"
        assert 'descriptions' in train_batch, "Batch should contain descriptions"
        
        images = train_batch['images']
        descriptions = train_batch['descriptions']
        
        assert images.shape[-1] == 215, "Images should be 215x215"
        assert images.shape[-2] == 215, "Images should be 215x215"
        assert len(descriptions) == images.shape[0], "Should have same number of descriptions as images"
    
    print("✓ Data Loading test passed")


def main():
    """Run all tests."""
    print("Running comprehensive tests for Pokemon Sprite Generator...")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        test_text_encoder()
        test_cnn_decoder()
        test_attention_mechanism()
        test_complete_model()
        test_model_save_load()
        test_configuration()
        test_data_loading()
        
        print("\n" + "=" * 60)
        print("🎉 All comprehensive tests passed successfully!")
        print("The Pokemon Sprite Generator is fully functional!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)