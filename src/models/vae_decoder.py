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


class AttentionBlock(nn.Module):
    """Multi-head self-attention block for spatial features."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(1)
        
        # Compute attention
        attn = torch.softmax(q.transpose(-2, -1) @ k / math.sqrt(self.head_dim), dim=-1)
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        
        x = x.reshape(b, c, h, w)
        x = self.proj(x)
        
        return x + residual


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
    """VAE encoder that encodes Pokemon images to latent space."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            # 215x215 -> 107x107
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(64, 64),
            
            # 107x107 -> 53x53
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(128, 128),
            
            # 53x53 -> 26x26
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(256, 256),
            
            # 26x26 -> 13x13
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(512, 512),
            
            # 13x13 -> 6x6
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResNetBlock(512, 512),
            
            # 6x6 -> 3x3
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Latent space projections
        self.mu_proj = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)
        self.logvar_proj = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image to latent space.
        
        Args:
            x: Input image [batch_size, 3, 215, 215]
            
        Returns:
            latent: Sampled latent vector [batch_size, latent_dim, 3, 3]
            mu: Mean of latent distribution [batch_size, latent_dim, 3, 3]
            logvar: Log variance of latent distribution [batch_size, latent_dim, 3, 3]
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
    """VAE decoder that generates Pokemon images from latent space with text conditioning."""
    
    def __init__(self, latent_dim: int = 512, text_dim: int = 256, output_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        
        # Initial latent processing
        self.latent_proj = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1)
        
        # Decoder blocks with cross-attention - store layers individually
        # Block 1: 3x3 -> 6x6
        self.block1_resnet1 = ResNetBlock(512, 512)
        self.block1_attn = CrossAttentionBlock(512, text_dim)
        self.block1_resnet2 = ResNetBlock(512, 512)
        self.block1_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Block 2: 6x6 -> 13x13
        self.block2_resnet1 = ResNetBlock(512, 512)
        self.block2_attn = CrossAttentionBlock(512, text_dim)
        self.block2_resnet2 = ResNetBlock(512, 512)
        self.block2_upsample = nn.Upsample(size=(13, 13), mode='bilinear', align_corners=False)
        
        # Block 3: 13x13 -> 26x26
        self.block3_resnet1 = ResNetBlock(512, 256)
        self.block3_attn = CrossAttentionBlock(256, text_dim)
        self.block3_resnet2 = ResNetBlock(256, 256)
        self.block3_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Block 4: 26x26 -> 53x53
        self.block4_resnet1 = ResNetBlock(256, 128)
        self.block4_attn = CrossAttentionBlock(128, text_dim)
        self.block4_resnet2 = ResNetBlock(128, 128)
        self.block4_upsample = nn.Upsample(size=(53, 53), mode='bilinear', align_corners=False)
        
        # Block 5: 53x53 -> 107x107
        self.block5_resnet1 = ResNetBlock(128, 64)
        self.block5_attn = CrossAttentionBlock(64, text_dim)
        self.block5_resnet2 = ResNetBlock(64, 64)
        self.block5_upsample = nn.Upsample(size=(107, 107), mode='bilinear', align_corners=False)
        
        # Block 6: 107x107 -> 215x215
        self.block6_resnet1 = ResNetBlock(64, 32)
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
            latent: Latent vector [batch_size, latent_dim, 3, 3]
            text_emb: Text embeddings [batch_size, seq_len, text_dim]
            
        Returns:
            Generated image [batch_size, 3, 215, 215]
        """
        x = self.latent_proj(latent)
        
        # Pass through decoder blocks with cross-attention
        # Block 1: 3x3 -> 6x6
        x = self.block1_resnet1(x)
        x = self.block1_attn(x, text_emb)
        x = self.block1_resnet2(x)
        x = self.block1_upsample(x)
        
        # Block 2: 6x6 -> 13x13
        x = self.block2_resnet1(x)
        x = self.block2_attn(x, text_emb)
        x = self.block2_resnet2(x)
        x = self.block2_upsample(x)
        
        # Block 3: 13x13 -> 26x26
        x = self.block3_resnet1(x)
        x = self.block3_attn(x, text_emb)
        x = self.block3_resnet2(x)
        x = self.block3_upsample(x)
        
        # Block 4: 26x26 -> 53x53
        x = self.block4_resnet1(x)
        x = self.block4_attn(x, text_emb)
        x = self.block4_resnet2(x)
        x = self.block4_upsample(x)
        
        # Block 5: 53x53 -> 107x107
        x = self.block5_resnet1(x)
        x = self.block5_attn(x, text_emb)
        x = self.block5_resnet2(x)
        x = self.block5_upsample(x)
        
        # Block 6: 107x107 -> 215x215
        x = self.block6_resnet1(x)
        x = self.block6_attn(x, text_emb)
        x = self.block6_resnet2(x)
        x = self.block6_upsample(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


class NoiseScheduler:
    """Noise scheduler for diffusion-like training."""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean latents."""
        # Ensure tensors are on the same device
        device = x_0.device
        if self.alphas_cumprod.device != device:
            self.to(device)
            
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
    def to(self, device):
        """Move scheduler to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self


class PokemonVAE(nn.Module):
    """Complete VAE model for Pokemon sprite generation."""
    
    def __init__(self, latent_dim: int = 512, text_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        
        self.encoder = VAEEncoder(input_channels=3, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, text_dim=text_dim, output_channels=3)
        self.noise_scheduler = NoiseScheduler()
        
    def forward(self, images: torch.Tensor, text_emb: torch.Tensor, mode: str = 'train') -> dict:
        """
        Forward pass of the VAE.
        
        Args:
            images: Input images [batch_size, 3, 215, 215]
            text_emb: Text embeddings [batch_size, seq_len, text_dim]
            mode: 'train' or 'inference'
            
        Returns:
            Dictionary with outputs and losses
        """
        if mode == 'train':
            # Encode images to latent space
            latent, mu, logvar = self.encoder(images)
            
            # Add noise for diffusion-like training
            noise = torch.randn_like(latent)
            timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, (latent.size(0),), device=latent.device)
            noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
            
            # Decode with text conditioning
            reconstructed = self.decoder(noisy_latent, text_emb)
            
            # Compute losses
            recon_loss = F.mse_loss(reconstructed, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            return {
                'reconstructed': reconstructed,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'latent': latent,
                'mu': mu,
                'logvar': logvar
            }
        
        else:  # inference
            # Sample from prior
            batch_size = text_emb.size(0)
            latent = torch.randn(batch_size, self.latent_dim, 3, 3, device=text_emb.device)
            
            # Decode with text conditioning
            generated = self.decoder(latent, text_emb)
            
            return {
                'generated': generated,
                'latent': latent
            }
    
    def to_device(self, device):
        """Move model to device."""
        super().to(device)
        self.noise_scheduler.to(device)
        return self


class DiffusionVAEDecoder(nn.Module):
    """Diffusion-style VAE decoder that learns to denoise latent representations."""
    
    def __init__(self, latent_dim: int = 512, text_dim: int = 256, output_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        
        # Time embedding for diffusion steps
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        # Main decoder with time and text conditioning
        self.decoder = VAEDecoder(latent_dim, text_dim, output_channels)
        
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int = 128) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
        
    def forward(self, noisy_latent: torch.Tensor, text_emb: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Denoise latent vector with text conditioning.
        
        Args:
            noisy_latent: Noisy latent vector [batch_size, latent_dim, 3, 3]
            text_emb: Text embeddings [batch_size, seq_len, text_dim]
            timesteps: Timestep indices [batch_size]
            
        Returns:
            Denoised image [batch_size, 3, 215, 215]
        """
        # Get time embeddings
        time_emb = self.time_embed(self.get_timestep_embedding(timesteps))
        
        # Add time information to text embeddings
        time_emb = time_emb.unsqueeze(1).expand(-1, text_emb.size(1), -1)
        conditioned_text = text_emb + time_emb
        
        # Decode with conditioning
        return self.decoder(noisy_latent, conditioned_text)
