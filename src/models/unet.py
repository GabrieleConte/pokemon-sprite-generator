"""
U-Net architecture for denoising latent representations in Pokemon sprite generation.
Based on Stable Diffusion's U-Net with text conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding layer that converts timesteps to high-dimensional embeddings.
    """
    
    def __init__(self, embedding_dim: int = 128, max_time: int = 1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_time = max_time
        
        # Sinusoidal embeddings
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb_coeff', torch.exp(torch.arange(half_dim) * -emb))
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert timesteps to embeddings.
        
        Args:
            timesteps: [batch_size] timestep indices
            
        Returns:
            time_emb: [batch_size, embedding_dim] time embeddings
        """
        # Create sinusoidal embeddings
        timesteps = timesteps.float()
        emb_coeff = self.emb_coeff.to(timesteps.device)
        emb = timesteps.unsqueeze(-1) * emb_coeff.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Process through MLP
        time_emb = self.time_mlp(emb)
        
        return time_emb


class ResBlock(nn.Module):
    """
    Residual block with time and text conditioning.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 128,
                 text_emb_dim: int = 256, groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time conditioning
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Text conditioning  
        self.text_proj = nn.Linear(text_emb_dim, out_channels)
        
        # Second conv block
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time and text conditioning.
        
        Args:
            x: [batch_size, in_channels, height, width] input features
            time_emb: [batch_size, time_emb_dim] time embeddings
            text_emb: [batch_size, text_emb_dim] text embeddings (pooled)
            
        Returns:
            output: [batch_size, out_channels, height, width] output features
        """
        residual = x
        
        # First conv block
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        
        # Add time conditioning
        time_proj = self.time_proj(time_emb)[:, :, None, None]
        x = x + time_proj
        
        # Add text conditioning
        text_proj = self.text_proj(text_emb)[:, :, None, None]
        x = x + text_proj
        
        # Second conv block
        x = F.silu(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        
        # Skip connection
        return x + self.skip_conv(residual)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for text conditioning in U-Net.
    """
    
    def __init__(self, channels: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Layer normalization
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Text projection
        self.text_proj = nn.Linear(text_dim, channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention and cross-attention.
        
        Args:
            x: [batch_size, channels, height, width] input features
            text_emb: [batch_size, seq_len, text_dim] text embeddings
            
        Returns:
            output: [batch_size, channels, height, width] output features
        """
        batch_size, channels, height, width = x.shape
        
        # Reshape for attention
        x_flat = x.view(batch_size, channels, height * width).permute(0, 2, 1)  # [B, HW, C]
        
        # Self-attention
        residual = x_flat
        x_norm = self.norm1(x_flat.permute(0, 2, 1)).permute(0, 2, 1)  # Group norm expects [B, C, HW]
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_flat = residual + x_attn
        
        # Cross-attention with text
        residual = x_flat
        x_norm = self.norm2(x_flat.permute(0, 2, 1)).permute(0, 2, 1)  # Group norm expects [B, C, HW]
        text_proj = self.text_proj(text_emb)  # [B, seq_len, C]
        x_cross, _ = self.cross_attn(x_norm, text_proj, text_proj)
        x_flat = residual + x_cross
        
        # Feed-forward
        residual = x_flat
        x_ff = self.ffn(x_flat)
        x_flat = residual + x_ff
        
        # Reshape back
        x_out = x_flat.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        return x_out


class UNetBlock(nn.Module):
    """
    U-Net block combining ResBlock and CrossAttentionBlock.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 128,
                 text_emb_dim: int = 256, has_attention: bool = True, num_heads: int = 8):
        super().__init__()
        self.has_attention = has_attention
        
        # Residual block
        self.res_block = ResBlock(in_channels, out_channels, time_emb_dim, text_emb_dim)
        
        # Cross-attention block
        if has_attention:
            self.attn_block = CrossAttentionBlock(out_channels, text_emb_dim, num_heads)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor,
                text_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net block.
        
        Args:
            x: [batch_size, in_channels, height, width] input features
            time_emb: [batch_size, time_emb_dim] time embeddings
            text_emb: [batch_size, text_emb_dim] pooled text embeddings
            text_seq: [batch_size, seq_len, text_emb_dim] sequential text embeddings
            
        Returns:
            output: [batch_size, out_channels, height, width] output features
        """
        # Apply residual block
        x = self.res_block(x, time_emb, text_emb)
        
        # Apply cross-attention if enabled
        if self.has_attention:
            x = self.attn_block(x, text_seq)
            
        return x


class UNet(nn.Module):
    """
    U-Net for denoising latent representations with text conditioning.
    Based on Stable Diffusion's U-Net architecture.
    """
    
    def __init__(self, latent_dim: int = 1024, text_dim: int = 256, 
                 time_emb_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.time_emb_dim = time_emb_dim
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_emb_dim)
        
        # Text pooling for conditioning
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initial projection - reduce channels from 1024 to manageable size
        self.init_conv = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1)
        
        # Encoder blocks - working at 4x4 resolution
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(512, 512, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(512, 512, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(512, 512, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        
        # Middle block
        self.middle_block = UNetBlock(512, 512, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads)
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(512 + 512, 512, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(512 + 512, 512, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(512 + 512, 512, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        
        # Final projection back to latent dimension
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor, 
                text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            noisy_latent: [batch_size, latent_dim, 3, 3] noisy latent representation
            timesteps: [batch_size] timestep indices
            text_emb: [batch_size, seq_len, text_dim] text embeddings
            
        Returns:
            denoised_latent: [batch_size, latent_dim, 3, 3] denoised latent representation
        """
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Pool text embeddings for conditioning
        text_pooled = self.text_pool(text_emb.transpose(1, 2)).squeeze(-1)  # [B, text_dim]
        
        # Initial convolution
        x = self.init_conv(noisy_latent)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder path - collect skip connections
        skip_connections = []
        
        for block in self.encoder_blocks:
            x = block(x, time_emb, text_pooled, text_emb)
            skip_connections.append(x)
        
        # Middle processing
        x = self.middle_block(x, time_emb, text_pooled, text_emb)
        
        # Decoder path - use skip connections in reverse order
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i+1)]  # Reverse order
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_emb, text_pooled, text_emb)
        
        # Final projection
        denoised_latent = self.final_conv(x)
        
        return denoised_latent


def test_unet():
    """Test function for U-Net architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    unet = UNet(latent_dim=1024, text_dim=256).to(device)
    
    # Create test inputs
    batch_size = 4
    noisy_latent = torch.randn(batch_size, 1024, 4, 4).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    text_emb = torch.randn(batch_size, 32, 256).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = unet(noisy_latent, timesteps, text_emb)
    
    print(f"Input shape: {noisy_latent.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    assert output.shape == noisy_latent.shape, "Output shape should match input shape"
    print("U-Net test passed!")


if __name__ == "__main__":
    test_unet()
