import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class ResNetBlock(nn.Module):
    """ResNet block with group normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + self.shortcut(residual)

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for text conditioning."""
    
    def __init__(self, channels: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Linear(text_dim, channels)
        self.v = nn.Linear(text_dim, channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        q = self.q(x).reshape(b, self.num_heads, self.head_dim, h * w)
        k = self.k(text_emb).reshape(b, self.num_heads, self.head_dim, -1)
        v = self.v(text_emb).reshape(b, self.num_heads, self.head_dim, -1)
        
        # Compute cross-attention
        attn = torch.softmax(q.transpose(-2, -1) @ k / math.sqrt(self.head_dim), dim=-1)
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        
        x = x.reshape(b, c, h, w)
        x = self.proj(x)
        
        return x + residual


class VAEEncoder(nn.Module):
    """VAE encoder that encodes Pokemon images to latent space with gradual downsampling."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # More gradual encoder network: 215x215 -> 4x4
        self.encoder = nn.Sequential(
            # 215x215 -> 107x107
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(32, 32),
            
            # 107x107 -> 53x53
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(64, 64),
            
            # 53x53 -> 26x26
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(128, 128),
            
            # 26x26 -> 13x13
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(256, 256),
            
            # 13x13 -> 6x6
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(512, 512),
            
            # 6x6 -> 4x4
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            ResNetBlock(512, 512),
        )
        
        # Latent space projections for 4x4 output
        self.mu_proj = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)
        self.logvar_proj = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image to latent space.
        
        Args:
            x: Input image [batch_size, 3, 215, 215]
            
        Returns:
            latent: Sampled latent vector [batch_size, latent_dim, 4, 4]
            mu: Mean of latent distribution [batch_size, latent_dim, 4, 4]
            logvar: Log variance of latent distribution [batch_size, latent_dim, 4, 4]
        """
        encoded = self.encoder(x)
        
        mu = self.mu_proj(encoded)
        logvar = self.logvar_proj(encoded)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        
        return latent, mu, logvar


class VAEDecoder(nn.Module):
    """VAE decoder that generates Pokemon images from latent space with text conditioning and gradual upsampling."""
    
    def __init__(self, latent_dim: int = 1024, text_dim: int = 256, output_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        
        # Initial latent processing - reduce to manageable channels
        self.latent_proj = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1)
        
        # Gradual decoder blocks with cross-attention: 4x4 -> 215x215
        # Block 1: 4x4 -> 7x7
        self.block1_resnet1 = ResNetBlock(512, 512)
        self.block1_attn = CrossAttentionBlock(512, text_dim)
        self.block1_resnet2 = ResNetBlock(512, 512)
        self.block1_upsample = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False)
        
        # Block 2: 7x7 -> 14x14
        self.block2_resnet1 = ResNetBlock(512, 256)
        self.block2_attn = CrossAttentionBlock(256, text_dim)
        self.block2_resnet2 = ResNetBlock(256, 256)
        self.block2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Block 3: 14x14 -> 27x27
        self.block3_resnet1 = ResNetBlock(256, 128)
        self.block3_attn = CrossAttentionBlock(128, text_dim)
        self.block3_resnet2 = ResNetBlock(128, 128)
        self.block3_upsample = nn.Upsample(size=(27, 27), mode='bilinear', align_corners=False)
        
        # Block 4: 27x27 -> 54x54
        self.block4_resnet1 = ResNetBlock(128, 64)
        self.block4_attn = CrossAttentionBlock(64, text_dim)
        self.block4_resnet2 = ResNetBlock(64, 64)
        self.block4_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Block 5: 54x54 -> 108x108
        self.block5_resnet1 = ResNetBlock(64, 32)
        self.block5_attn = CrossAttentionBlock(32, text_dim)
        self.block5_resnet2 = ResNetBlock(32, 32)
        self.block5_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Block 6: 108x108 -> 215x215
        self.block6_resnet1 = ResNetBlock(32, 32)
        self.block6_attn = CrossAttentionBlock(32, text_dim)
        self.block6_resnet2 = ResNetBlock(32, 32)
        self.block6_upsample = nn.Upsample(size=(215, 215), mode='bilinear', align_corners=False)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, latent: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image with text conditioning.
        
        Args:
            latent: Latent vector [batch_size, latent_dim, 4, 4]
            text_emb: Text embeddings [batch_size, seq_len, text_dim]
            
        Returns:
            image: Generated image [batch_size, 3, 215, 215]
        """
        # Project latent to initial feature map
        x = self.latent_proj(latent)
        
        # Progressive upsampling with text conditioning
        # Block 1: 4x4 -> 7x7
        x = self.block1_resnet1(x)
        x = self.block1_attn(x, text_emb)
        x = self.block1_resnet2(x)
        x = self.block1_upsample(x)
        
        # Block 2: 7x7 -> 14x14
        x = self.block2_resnet1(x)
        x = self.block2_attn(x, text_emb)
        x = self.block2_resnet2(x)
        x = self.block2_upsample(x)
        
        # Block 3: 14x14 -> 27x27
        x = self.block3_resnet1(x)
        x = self.block3_attn(x, text_emb)
        x = self.block3_resnet2(x)
        x = self.block3_upsample(x)
        
        # Block 4: 27x27 -> 54x54
        x = self.block4_resnet1(x)
        x = self.block4_attn(x, text_emb)
        x = self.block4_resnet2(x)
        x = self.block4_upsample(x)
        
        # Block 5: 54x54 -> 108x108
        x = self.block5_resnet1(x)
        x = self.block5_attn(x, text_emb)
        x = self.block5_resnet2(x)
        x = self.block5_upsample(x)
        
        # Block 6: 108x108 -> 215x215
        x = self.block6_resnet1(x)
        x = self.block6_attn(x, text_emb)
        x = self.block6_resnet2(x)
        x = self.block6_upsample(x)
        
        # Final output
        image = self.final_conv(x)
        
        return image


class PokemonVAE(nn.Module):
    """Complete VAE model for Pokemon sprite generation."""
    
    def __init__(self, latent_dim: int = 1024, text_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        
        self.encoder = VAEEncoder(input_channels=3, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, text_dim=text_dim, output_channels=3)
        
    def forward(self, images: torch.Tensor, text_emb: torch.Tensor, mode: str = 'train') -> dict:
        """
        Forward pass of the VAE.
        
        Args:
            images: Input images [batch_size, 3, 215, 215]
            text_emb: Text embeddings [batch_size, seq_len, text_dim]
            mode: 'train', 'val', or 'generate'
            
        Returns:
            Dictionary with outputs and losses
        """
        # Encode images to latent space
        latent, mu, logvar = self.encoder(images)
        
        if mode == 'generate':
            # Use only the mean for generation
            latent = mu
        
        # Decode latent to images
        reconstructed = self.decoder(latent, text_emb)
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'mu': mu,
            'logvar': logvar
        }
    
    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images to latent space."""
        return self.encoder(images)
    
    def decode(self, latent: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images."""
        return self.decoder(latent, text_emb)
    
    def sample(self, batch_size: int, text_emb: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sample new images from random latent vectors."""
        # Sample random latent vectors
        latent = torch.randn(batch_size, self.latent_dim, 4, 4, device=device)
        
        # Decode to images
        return self.decoder(latent, text_emb)


def test_vae():
    """Test function for VAE architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    vae = PokemonVAE(latent_dim=1024, text_dim=256).to(device)
    
    # Create test inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 215, 215).to(device)
    text_emb = torch.randn(batch_size, 32, 256).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = vae(images, text_emb, mode='train')
    
    print(f"Input shape: {images.shape}")
    print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"Latent shape: {outputs['latent'].shape}")
    print(f"Mu shape: {outputs['mu'].shape}")
    print(f"Logvar shape: {outputs['logvar'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    assert outputs['reconstructed'].shape == images.shape, "Reconstructed shape should match input"
    assert outputs['latent'].shape == (batch_size, 1024, 4, 4), "Latent should be [batch, 1024, 4, 4]"
    print("VAE test passed!")


if __name__ == "__main__":
    test_vae()