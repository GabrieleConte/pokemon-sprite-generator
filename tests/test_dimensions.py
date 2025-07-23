#!/usr/bin/env python3

import torch
from src.models.vae_decoder import VAEEncoder, VAEDecoder, PokemonVAE
from src.models.unet import UNet

def test_vae_dimensions():
    print("Testing VAE dimensions...")
    
    # Test encoder
    encoder = VAEEncoder(latent_dim=8)
    x = torch.randn(2, 3, 215, 215)
    latent, mu, logvar = encoder(x)
    print(f"Encoder - Input: {x.shape}")
    print(f"Encoder - Latent: {latent.shape}")
    print(f"Expected latent: [2, 8, 27, 27]")
    print(f"Latent shape correct: {latent.shape == torch.Size([2, 8, 27, 27])}")
    
    # Test decoder
    decoder = VAEDecoder(latent_dim=8, text_dim=256)
    text_emb = torch.randn(2, 32, 256)
    reconstructed = decoder(latent, text_emb)
    print(f"Decoder - Latent: {latent.shape}")
    print(f"Decoder - Output: {reconstructed.shape}")
    print(f"Expected output: [2, 3, 215, 215]")
    print(f"Output shape correct: {reconstructed.shape == torch.Size([2, 3, 215, 215])}")
    
    # Test full VAE
    vae = PokemonVAE(latent_dim=8, text_dim=256)
    outputs = vae(x, text_emb, mode='train')
    print(f"Full VAE - Reconstructed: {outputs['reconstructed'].shape}")
    print(f"Full VAE - Latent: {outputs['latent'].shape}")
    
    print(f"VAE Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    return latent.shape == torch.Size([2, 8, 27, 27])

def test_unet_dimensions():
    print("\nTesting U-Net dimensions...")
    
    unet = UNet(latent_dim=8, text_dim=256)
    noisy_latent = torch.randn(2, 8, 27, 27)
    timesteps = torch.randint(0, 1000, (2,))
    text_emb = torch.randn(2, 32, 256)
    
    output = unet(noisy_latent, timesteps, text_emb)
    print(f"U-Net - Input: {noisy_latent.shape}")
    print(f"U-Net - Output: {output.shape}")
    print(f"Output shape correct: {output.shape == noisy_latent.shape}")
    
    print(f"U-Net Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    return output.shape == noisy_latent.shape

if __name__ == "__main__":
    vae_ok = test_vae_dimensions()
    unet_ok = test_unet_dimensions()
    
    print(f"\nResults:")
    print(f"VAE Test: {'PASSED' if vae_ok else 'FAILED'}")
    print(f"U-Net Test: {'PASSED' if unet_ok else 'FAILED'}")
    
    if vae_ok and unet_ok:
        print("All tests PASSED! 🎉")
    else:
        print("Some tests FAILED! ❌")
