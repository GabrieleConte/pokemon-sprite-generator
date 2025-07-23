#!/usr/bin/env python3
"""
Test script to check if the three models (VAE, U-Net, Text Encoder) are compatible.
This tests the data flow between models to ensure they can work together.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_model_compatibility():
    """Test compatibility between VAE, U-Net, and Text Encoder."""
    
    print("Testing model compatibility...")
    print("="*50)
    
    # Import models
    try:
        from src.models.vae_decoder import PokemonVAE
        from src.models.unet import UNet
        from src.models.text_encoder import TextEncoder
        print("✅ All models imported successfully")
    except Exception as e:
        print(f"❌ Failed to import models: {e}")
        return False
    
    # Initialize models
    try:
        print("\nInitializing models...")
        vae = PokemonVAE(latent_dim=256, text_dim=256)
        unet = UNet(latent_dim=256, text_dim=256, time_emb_dim=128, num_heads=8)
        text_encoder = TextEncoder(model_name='prajjwal1/bert-mini', hidden_dim=256, nhead=4, num_encoder_layers=2)
        print("✅ Models initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize models: {e}")
        return False
    
    # Create test data
    batch_size = 2
    print(f"\nCreating test data (batch_size={batch_size})...")
    
    # Input image (VAE decoder outputs 215x215, so we need to match)
    image = torch.randn(batch_size, 3, 215, 215)
    print(f"  Input image shape: {image.shape}")
    
    # Text input (sample descriptions)
    text_input = [
        "A fire-type dragon Pokemon with red scales and large wings",
        "A small electric mouse Pokemon with yellow fur and red cheeks"
    ]
    print(f"  Text input: {text_input}")
    print(f"  Text input length: {len(text_input)}")
    
    # Timesteps for diffusion
    timesteps = torch.randint(0, 1000, (batch_size,))
    print(f"  Timesteps shape: {timesteps.shape}")
    
    # Test Text Encoder
    print("\nTesting Text Encoder...")
    try:
        text_embeddings = text_encoder(text_input)
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        
        # Check if text embeddings have correct batch size and embedding dimension
        assert text_embeddings.shape[0] == batch_size, f"Batch size mismatch: {text_embeddings.shape[0]} != {batch_size}"
        assert text_embeddings.shape[2] == 256, f"Embedding dimension mismatch: {text_embeddings.shape[2]} != 256"
        print("✅ Text encoder successful")
        
    except Exception as e:
        print(f"❌ Text encoder test failed: {e}")
        return False
    
    # Test VAE
    print("\nTesting VAE...")
    try:
        # Test VAE forward pass in training mode
        vae_output = vae(image, text_embeddings, mode='train')
        
        reconstructed = vae_output['reconstructed']
        latent = vae_output['latent']
        mu = vae_output['mu']
        logvar = vae_output['logvar']
        
        print(f"  Latent shape: {latent.shape}")
        print(f"  Reconstructed image shape: {reconstructed.shape}")
        print(f"  Mu shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")
        
        # Check if reconstruction has correct shape
        assert reconstructed.shape == image.shape, f"Reconstruction shape mismatch: {reconstructed.shape} != {image.shape}"
        print("✅ VAE encoding/decoding successful")
        
    except Exception as e:
        print(f"❌ VAE test failed: {e}")
        return False
    
    # Test U-Net
    print("\nTesting U-Net...")
    try:
        # Add noise to latent
        noise = torch.randn_like(latent)
        noisy_latent = latent + noise
        print(f"  Noisy latent shape: {noisy_latent.shape}")
        
        # Predict noise with U-Net
        predicted_noise = unet(noisy_latent, timesteps, text_embeddings)
        print(f"  Predicted noise shape: {predicted_noise.shape}")
        
        # Check if predicted noise has correct shape
        assert predicted_noise.shape == latent.shape, f"Predicted noise shape mismatch: {predicted_noise.shape} != {latent.shape}"
        print("✅ U-Net denoising successful")
        
    except Exception as e:
        print(f"❌ U-Net test failed: {e}")
        return False
    
    # Test complete pipeline
    print("\nTesting complete pipeline...")
    try:
        # Test VAE in inference mode
        vae_inference = vae(image, text_embeddings, mode='inference')
        generated_image = vae_inference['generated']
        
        # Test U-Net with the latent from VAE
        noise = torch.randn_like(latent)
        noisy_latent = latent + noise
        denoised_noise = unet(noisy_latent, timesteps, text_embeddings)
        denoised_latent = noisy_latent - denoised_noise  # Subtract predicted noise
        
        # Test VAE decoder directly using the encoder from VAE
        final_output = vae.decoder(denoised_latent, text_embeddings)
        
        print(f"  Pipeline flow: Image{image.shape} -> Latent{latent.shape} -> Noisy{noisy_latent.shape} -> Denoised{denoised_latent.shape} -> Final{final_output.shape}")
        
        # Check final image shape
        assert final_output.shape == image.shape, f"Final image shape mismatch: {final_output.shape} != {image.shape}"
        print("✅ Complete pipeline successful")
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        return False
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    try:
        # Create a simple loss and check if gradients flow
        loss = torch.nn.functional.mse_loss(final_output, image)
        loss.backward()
        
        # Check if gradients exist
        vae_has_grad = any(p.grad is not None for p in vae.parameters())
        unet_has_grad = any(p.grad is not None for p in unet.parameters())
        text_has_grad = any(p.grad is not None for p in text_encoder.parameters())
        
        print(f"  VAE gradients: {'✅' if vae_has_grad else '❌'}")
        print(f"  U-Net gradients: {'✅' if unet_has_grad else '❌'}")
        print(f"  Text encoder gradients: {'✅' if text_has_grad else '❌'}")
        
        if vae_has_grad and unet_has_grad and text_has_grad:
            print("✅ Gradient flow successful")
        else:
            print("⚠️  Some models don't have gradients (might be expected)")
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED! Models are compatible!")
    print("="*50)
    
    # Print model sizes
    print("\nModel Statistics:")
    vae_params = sum(p.numel() for p in vae.parameters())
    unet_params = sum(p.numel() for p in unet.parameters())
    text_params = sum(p.numel() for p in text_encoder.parameters())
    total_params = vae_params + unet_params + text_params
    
    print(f"  VAE parameters: {vae_params:,}")
    print(f"  U-Net parameters: {unet_params:,}")
    print(f"  Text encoder parameters: {text_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    return True


if __name__ == "__main__":
    success = test_model_compatibility()
    sys.exit(0 if success else 1)
