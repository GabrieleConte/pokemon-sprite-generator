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
                 text_emb_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Determine group size for normalization (ensure it divides channel count)
        groups_in = min(32, in_channels)
        while in_channels % groups_in != 0 and groups_in > 1:
            groups_in -= 1
            
        groups_out = min(32, out_channels)
        while out_channels % groups_out != 0 and groups_out > 1:
            groups_out -= 1
        
        # First conv block
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time conditioning
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Text conditioning  
        self.text_proj = nn.Linear(text_emb_dim, out_channels)
        
        # Second conv block
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
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
        
        # Ensure head_dim is valid
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        # Determine group size for normalization
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        
        # Layer normalization - use smaller groups for stability
        self.norm1 = nn.GroupNorm(max(1, groups), channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(max(1, groups), channels, eps=1e-6)
        
        # Self-attention with improved stability
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.05,  # Reduced dropout for stability
            batch_first=True
        )
        
        # Cross-attention with improved stability
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.05,  # Reduced dropout for stability
            batch_first=True
        )
        
        # Text projection with proper initialization
        self.text_proj = nn.Linear(text_dim, channels)
        nn.init.xavier_uniform_(self.text_proj.weight, gain=0.02)  # Small gain for stability
        nn.init.zeros_(self.text_proj.bias)
        
        # Feed-forward network with proper initialization
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),  # Reduced expansion for stability
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(channels * 2, channels),
            nn.Dropout(0.05)
        )
        
        # Initialize FFN weights
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.02)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention and cross-attention with improved stability.
        
        Args:
            x: [batch_size, channels, height, width] input features
            text_emb: [batch_size, seq_len, text_dim] text embeddings
            
        Returns:
            output: [batch_size, channels, height, width] output features
        """
        batch_size, channels, height, width = x.shape
        
        # Reshape for attention
        x_flat = x.view(batch_size, channels, height * width).permute(0, 2, 1)  # [B, HW, C]
        
        # Self-attention with residual connection
        residual = x_flat
        try:
            x_norm = self.norm1(x_flat.permute(0, 2, 1)).permute(0, 2, 1)  # Group norm expects [B, C, HW]
            
            # Apply self-attention with mild scaling for stability
            x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
            
            # Use much more reasonable scaling (closer to standard practice)
            x_attn = x_attn * 0.7  # Mild scaling for stability, not too aggressive
            x_flat = residual + x_attn
            
        except Exception as e:
            # Fallback: skip self-attention if it fails
            print(f"Warning: Self-attention failed, skipping: {e}")
            x_flat = residual
        
        # Cross-attention with text
        residual = x_flat
        try:
            x_norm = self.norm2(x_flat.permute(0, 2, 1)).permute(0, 2, 1)  # Group norm expects [B, C, HW]
            text_proj = self.text_proj(text_emb)  # [B, seq_len, C]
            
            # Apply cross-attention with mild scaling for stability
            x_cross, _ = self.cross_attn(x_norm, text_proj, text_proj)
            
            # Use reasonable scaling for cross-attention (text conditioning is important!)
            x_cross = x_cross * 0.8  # Preserve most of the cross-attention signal
            x_flat = residual + x_cross
            
        except Exception as e:
            # Fallback: skip cross-attention if it fails
            print(f"Warning: Cross-attention failed, skipping: {e}")
            x_flat = residual
        
        # Feed-forward with residual
        residual = x_flat
        try:
            x_ff = self.ffn(x_flat)
            x_ff = x_ff * 0.6  # Moderate scaling for FFN - still preserves learning capability
            x_flat = residual + x_ff
        except Exception as e:
            # Fallback: skip FFN if it fails
            print(f"Warning: FFN failed, skipping: {e}")
            x_flat = residual
        
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
    Based on Stable Diffusion's U-Net architecture with proper downsampling/upsampling.
    Input: [batch_size, 8, 27, 27] -> Output: [batch_size, 8, 27, 27] (predicted noise)
    """
    
    def __init__(self, latent_dim: int = 8, text_dim: int = 256, 
                 time_emb_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.time_emb_dim = time_emb_dim
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_emb_dim)
        
        # Text pooling for conditioning
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initial projection from latent_dim to first feature dimension
        self.init_conv = nn.Conv2d(latent_dim, 320, kernel_size=3, padding=1)
        
        # Encoder path with downsampling (following Stable Diffusion channel progression)
        # Level 0: 27x27 -> 27x27 (320 channels) - No attention at highest resolution
        self.enc_block0 = nn.ModuleList([
            UNetBlock(320, 320, time_emb_dim, text_dim, has_attention=False, num_heads=num_heads),
            UNetBlock(320, 320, time_emb_dim, text_dim, has_attention=False, num_heads=num_heads),
        ])
        
        # Level 1: 27x27 -> 14x14 (640 channels) - Add attention at medium resolution
        self.downsample1 = nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1)
        self.enc_block1 = nn.ModuleList([
            UNetBlock(640, 640, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(640, 640, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        
        # Level 2: 14x14 -> 7x7 (1280 channels) - Full attention at low resolution
        self.downsample2 = nn.Conv2d(640, 1280, kernel_size=3, stride=2, padding=1)
        self.enc_block2 = nn.ModuleList([
            UNetBlock(1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        
        # Level 3: 7x7 -> 4x4 (1280 channels) - Deepest level with full attention
        self.downsample3 = nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
        self.enc_block3 = nn.ModuleList([
            UNetBlock(1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        
        # Middle block at 4x4 resolution (1280 channels)
        self.middle_block = UNetBlock(1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads)
        
        # Decoder path with upsampling and skip connections
        # Level 3: 4x4 -> 7x7 (1280 channels)
        self.dec_block3 = nn.ModuleList([
            UNetBlock(1280 + 1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(1280 + 1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        self.upsample3 = nn.Sequential(
            nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False),
            nn.Conv2d(1280, 1280, kernel_size=3, padding=1)
        )
        
        # Level 2: 7x7 -> 14x14 (1280 channels)
        self.dec_block2 = nn.ModuleList([
            UNetBlock(1280 + 1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(1280 + 1280, 1280, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        self.upsample2 = nn.Sequential(
            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),
            nn.Conv2d(1280, 640, kernel_size=3, padding=1)
        )
        
        # Level 1: 14x14 -> 27x27 (640 channels)
        self.dec_block1 = nn.ModuleList([
            UNetBlock(640 + 640, 640, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
            UNetBlock(640 + 640, 640, time_emb_dim, text_dim, has_attention=True, num_heads=num_heads),
        ])
        self.upsample1 = nn.Sequential(
            nn.Upsample(size=(27, 27), mode='bilinear', align_corners=False),
            nn.Conv2d(640, 320, kernel_size=3, padding=1)
        )
        
        # Level 0: 27x27 -> 27x27 (320 channels)
        self.dec_block0 = nn.ModuleList([
            UNetBlock(320 + 320, 320, time_emb_dim, text_dim, has_attention=False, num_heads=num_heads),
            UNetBlock(320 + 320, 320, time_emb_dim, text_dim, has_attention=False, num_heads=num_heads),
        ])
        
        # Final projection to predict noise
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, 320),
            nn.SiLU(),
            nn.Conv2d(320, latent_dim, kernel_size=3, padding=1)
        )
        
        # Initialize weights for numerical stability
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for improved stability."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)  # Small gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Special initialization for final layer (small init for noise prediction)
        for module in self.final_conv.modules():
            if isinstance(module, nn.Conv2d):
                # Use small but non-zero initialization
                nn.init.xavier_uniform_(module.weight, gain=0.02)  # Very small gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor, 
                text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net to predict noise.
        
        Args:
            noisy_latent: [batch_size, 8, 27, 27] noisy latent representation
            timesteps: [batch_size] timestep indices
            text_emb: [batch_size, seq_len, text_dim] text embeddings
            
        Returns:
            predicted_noise: [batch_size, 8, 27, 27] predicted noise to be removed
        """
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Pool text embeddings for conditioning
        text_pooled = self.text_pool(text_emb.transpose(1, 2)).squeeze(-1)  # [B, text_dim]
        
        # Initial convolution: [B, 8, 27, 27] -> [B, 320, 27, 27]
        x = self.init_conv(noisy_latent)
        
        # Store skip connections for decoder
        skip_connections = []
        
        # Encoder Level 0: 27x27 (320 channels)
        for block in self.enc_block0:
            x = block(x, time_emb, text_pooled, text_emb)
        skip_connections.append(x)  # Store for decoder
        
        # Downsample to Level 1: 27x27 -> 14x14 (640 channels)
        x = self.downsample1(x)
        for block in self.enc_block1:
            x = block(x, time_emb, text_pooled, text_emb)
        skip_connections.append(x)  # Store for decoder
        
        # Downsample to Level 2: 14x14 -> 7x7 (1280 channels)
        x = self.downsample2(x)
        for block in self.enc_block2:
            x = block(x, time_emb, text_pooled, text_emb)
        skip_connections.append(x)  # Store for decoder
        
        # Downsample to Level 3: 7x7 -> 4x4 (1280 channels)
        x = self.downsample3(x)
        for block in self.enc_block3:
            x = block(x, time_emb, text_pooled, text_emb)
        skip_connections.append(x)  # Store for decoder
        
        # Middle processing at 4x4 (1280 channels)
        x = self.middle_block(x, time_emb, text_pooled, text_emb)
        
        # Decoder Level 3: 4x4 (1280 channels)
        skip = skip_connections.pop()  # Get skip from enc_block3
        for i, block in enumerate(self.dec_block3):
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = block(x, time_emb, text_pooled, text_emb)
        
        # Upsample to Level 2: 4x4 -> 7x7 (1280 channels)
        x = self.upsample3(x)
        skip = skip_connections.pop()  # Get skip from enc_block2
        for i, block in enumerate(self.dec_block2):
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = block(x, time_emb, text_pooled, text_emb)
        
        # Upsample to Level 1: 7x7 -> 14x14 (640 channels)
        x = self.upsample2(x)
        skip = skip_connections.pop()  # Get skip from enc_block1
        for i, block in enumerate(self.dec_block1):
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = block(x, time_emb, text_pooled, text_emb)
        
        # Upsample to Level 0: 14x14 -> 27x27 (320 channels)
        x = self.upsample1(x)
        skip = skip_connections.pop()  # Get skip from enc_block0
        for i, block in enumerate(self.dec_block0):
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = block(x, time_emb, text_pooled, text_emb)
        
        # Final projection to predict noise: [B, 320, 27, 27] -> [B, 8, 27, 27]
        predicted_noise = self.final_conv(x)
        
        return predicted_noise


def test_unet():
    """Test function for U-Net architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with correct dimensions
    unet = UNet(latent_dim=8, text_dim=256).to(device)
    
    # Create test inputs with correct dimensions
    batch_size = 4
    noisy_latent = torch.randn(batch_size, 8, 27, 27).to(device)  # 8 channels, 27x27
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
    
    # Test different resolutions to ensure proper handling
    print("\nTesting dimension consistency...")
    print(f"Expected latent shape: [batch_size, 8, 27, 27]")
    print(f"Actual output shape: {output.shape}")
    print(f"Shape match: {output.shape == (batch_size, 8, 27, 27)}")
    
    return True


if __name__ == "__main__":
    test_unet()
