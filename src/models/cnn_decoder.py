import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    """CNN-based decoder for generating Pokemon sprites."""
    
    def __init__(self, latent_dim=256, noise_dim=100, image_size=215, 
                 num_channels=3, base_channels=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.num_channels = num_channels
        self.base_channels = base_channels
        
        # Total input dimension: latent (text) + noise
        self.input_dim = latent_dim + noise_dim
        
        # Calculate initial feature map size
        # We'll start with a small feature map and upsample to reach 215x215
        # 215 is close to 224 (7 * 32), so we'll use that as target
        self.init_size = 7  # 7x7 initial feature map
        self.init_channels = base_channels * 8  # 512 channels initially
        
        # Linear layer to project input to initial feature map
        self.fc = nn.Linear(self.input_dim, self.init_size * self.init_size * self.init_channels)
        
        # Transposed convolution layers to upsample
        self.conv_blocks = nn.ModuleList([
            # 7x7 -> 14x14
            self._make_conv_block(self.init_channels, base_channels * 4, kernel_size=4, stride=2, padding=1),
            # 14x14 -> 28x28  
            self._make_conv_block(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            # 28x28 -> 56x56
            self._make_conv_block(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            # 56x56 -> 112x112
            self._make_conv_block(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            # 112x112 -> 224x224 (close to target 215x215)
            self._make_conv_block(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
        ])
        
        # Final layer to get exact size and channels
        self.final_conv = nn.ConvTranspose2d(base_channels // 4, num_channels, 
                                           kernel_size=3, stride=1, padding=1)
        
        # Adaptive pooling to get exact target size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((image_size, image_size))
        
        # Initialize weights
        self._init_weights()
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        """Create a transposed convolution block with normalization and activation."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, text_features, noise=None):
        """
        Generate sprites from text features and noise.
        
        Args:
            text_features: Encoded text features [batch_size, latent_dim]
            noise: Random noise [batch_size, noise_dim] or None
            
        Returns:
            generated_images: Generated sprites [batch_size, channels, height, width]
        """
        batch_size = text_features.shape[0]
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=text_features.device)
        
        # Concatenate text features and noise
        combined_input = torch.cat([text_features, noise], dim=1)  # [batch_size, latent_dim + noise_dim]
        
        # Project to initial feature map
        x = self.fc(combined_input)  # [batch_size, init_size * init_size * init_channels]
        x = x.view(batch_size, self.init_channels, self.init_size, self.init_size)
        
        # Apply transposed convolution blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Final convolution to get RGB channels
        x = self.final_conv(x)
        
        # Adaptive pooling to get exact target size
        x = self.adaptive_pool(x)
        
        # Apply tanh activation to get pixel values in [-1, 1]
        x = torch.tanh(x)
        
        return x
    
    def generate_samples(self, text_features, num_samples=1):
        """Generate multiple samples for the same text features."""
        batch_size = text_features.shape[0]
        
        # Repeat text features for multiple samples
        text_features_expanded = text_features.unsqueeze(1).expand(-1, num_samples, -1)
        text_features_expanded = text_features_expanded.contiguous().view(-1, self.latent_dim)
        
        # Generate different noise for each sample
        noise = torch.randn(batch_size * num_samples, self.noise_dim, device=text_features.device)
        
        # Generate images
        generated_images = self.forward(text_features_expanded, noise)
        
        # Reshape back to [batch_size, num_samples, channels, height, width]
        generated_images = generated_images.view(batch_size, num_samples, self.num_channels, 
                                               self.image_size, self.image_size)
        
        return generated_images