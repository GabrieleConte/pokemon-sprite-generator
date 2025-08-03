#!/usr/bin/env python3
"""
Test script to verify BERT-base integration with VAE and U-Net components.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.text_encoder import TextEncoder
from src.models.vae_decoder import PokemonVAE
from src.models.diffusers_unet import DiffusersUNet
from optimize_mps_memory import setup_mps_memory_optimization




def test_text_encoder():
    """Test BERT-base text encoder."""
    print("Testing BERT-base Text Encoder...")
    
    # Create text encoder with BERT-base
    text_encoder = TextEncoder(
        model_name="google-bert/bert-base-uncased",
        hidden_dim=768
    )
    
    # Test input
    descriptions = [
        "A fire-type Pokemon with orange scales and a flame on its tail",
        "A water-type Pokemon with blue skin and large fins"
    ]
    
    # Forward pass
    with torch.no_grad():
        text_emb = text_encoder(descriptions)
    
    print(f"✅ Text embeddings shape: {text_emb.shape}")
    print(f"✅ Expected: [2, seq_len, 768]")
    assert text_emb.shape[0] == 2
    assert text_emb.shape[2] == 768
    print("✅ Text encoder test passed!")
    return text_emb


def test_vae_with_bert_base():
    """Test VAE with 768-dim text embeddings."""
    print("\nTesting VAE with BERT-base embeddings...")
    
    # Create VAE with 768-dim text embeddings
    vae = PokemonVAE(latent_dim=8, text_dim=768)
    
    # Test inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 215, 215)
    text_emb = torch.randn(batch_size, 32, 768)  # Simulated BERT-base embeddings
    
    # Forward pass
    with torch.no_grad():
        outputs = vae(images, text_emb, mode='train')
    
    print(f"✅ VAE input shape: {images.shape}")
    print(f"✅ VAE text embedding shape: {text_emb.shape}")
    print(f"✅ VAE output shape: {outputs['reconstructed'].shape}")
    print(f"✅ VAE latent shape: {outputs['latent'].shape}")
    
    assert outputs['reconstructed'].shape == images.shape
    assert outputs['latent'].shape == (batch_size, 8, 27, 27)
    print("✅ VAE test passed!")
    return outputs


def test_diffusers_unet_with_bert_base():
    """Test DiffusersUNet with 768-dim text embeddings."""
    print("\nTesting DiffusersUNet with BERT-base embeddings...")
    
    # Create U-Net with 768-dim text embeddings
    unet = DiffusersUNet(
        latent_dim=8,
        text_dim=768,  # BERT-base dimension
        cross_attention_dim=768,  # SD dimension (same as BERT-base)
        pretrained_model_name="runwayml/stable-diffusion-v1-5"
    )
    
    # Test inputs
    batch_size = 2
    noisy_latent = torch.randn(batch_size, 8, 27, 27)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 32, 768)  # BERT-base embeddings
    
    # Forward pass
    with torch.no_grad():
        noise_pred = unet(noisy_latent, timesteps, text_emb)
    
    print(f"✅ U-Net input shape: {noisy_latent.shape}")
    print(f"✅ U-Net text embedding shape: {text_emb.shape}")
    print(f"✅ U-Net output shape: {noise_pred.shape}")
    print(f"✅ U-Net trainable parameters: {unet.get_trainable_parameter_count():,}")
    
    assert noise_pred.shape == noisy_latent.shape
    print("✅ DiffusersUNet test passed!")
    return noise_pred


def test_end_to_end_integration():
    """Test complete pipeline with BERT-base."""
    print("\nTesting end-to-end integration...")
    
    # Create components
    text_encoder = TextEncoder(
        model_name="google-bert/bert-base-uncased",
        hidden_dim=768
    )
    
    vae = PokemonVAE(latent_dim=8, text_dim=768)
    
    unet = DiffusersUNet(
        latent_dim=8,
        text_dim=768,
        cross_attention_dim=768
    )
    
    # Test data
    descriptions = [
        "A fire-type Pokemon with orange scales",
        "A water-type Pokemon with blue skin"
    ]
    images = torch.randn(2, 3, 215, 215)
    
    # Complete pipeline
    with torch.no_grad():
        # 1. Encode text with BERT-large
        text_emb = text_encoder(descriptions)
        print(f"✅ Text encoding: {descriptions[0][:30]}... -> {text_emb.shape}")
        
        # 2. Encode images with VAE
        latent, mu, logvar = vae.encode(images)
        print(f"✅ Image encoding: {images.shape} -> {latent.shape}")
        
        # 3. Add noise (diffusion process)
        noise = torch.randn_like(latent)
        timesteps = torch.randint(0, 1000, (2,))
        # Simplified noise addition (normally done by noise scheduler)
        alpha = 0.5
        noisy_latent = alpha * latent + (1 - alpha) * noise
        
        # 4. Denoise with U-Net
        noise_pred = unet(noisy_latent, timesteps, text_emb)
        print(f"✅ Noise prediction: {noisy_latent.shape} -> {noise_pred.shape}")
        
        # 5. Decode back to image
        decoded_images = vae.decode(latent, text_emb)
        print(f"✅ Image decoding: {latent.shape} -> {decoded_images.shape}")
    
    print("🎉 End-to-end integration test passed!")
    print(f"🎉 Total model parameters:")
    print(f"   - Text Encoder: {sum(p.numel() for p in text_encoder.parameters()):,}")
    print(f"   - VAE: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"   - U-Net: {unet.get_parameter_count():,}")
    
    total_params = (sum(p.numel() for p in text_encoder.parameters()) + 
                   sum(p.numel() for p in vae.parameters()) + 
                   unet.get_parameter_count())
    print(f"   - Total: {total_params:,}")


def main():
    """Run all tests."""
    print("🧪 Testing BERT-base Integration")
    print("=" * 50)
    setup_mps_memory_optimization()  # Optimize MPS memory usage
    
    try:
        # Individual component tests
        test_text_encoder()
        test_vae_with_bert_base()
        test_diffusers_unet_with_bert_base()
        
        # End-to-end test
        test_end_to_end_integration()
        
        print("\n🎉 All tests passed! BERT-base integration is working correctly.")
        print("✅ Ready for training with 768-dimensional text embeddings!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
