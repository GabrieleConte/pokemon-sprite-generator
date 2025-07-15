import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer that connects CNN decoder features with transformer encoder outputs.
    Uses PyTorch's MultiheadAttention for proper attention computation.
    """
    def __init__(self, decoder_dim, encoder_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        
        # Use PyTorch's built-in MultiheadAttention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Linear projections to match dimensions
        self.key_proj = nn.Linear(encoder_dim, decoder_dim)
        self.value_proj = nn.Linear(encoder_dim, decoder_dim)
        
        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, decoder_features, encoder_output, encoder_mask=None):
        """
        Args:
            decoder_features: CNN decoder features [batch_size, spatial_dim, decoder_dim]
            encoder_output: Transformer encoder output [batch_size, seq_len, encoder_dim]
            encoder_mask: Optional mask for encoder output
        
        Returns:
            attended_features: Features after cross-attention [batch_size, spatial_dim, decoder_dim]
            attention_weights: Attention weights [batch_size, spatial_dim, seq_len]
        """
        # Project encoder outputs to decoder dimension
        keys = self.key_proj(encoder_output)  # [batch_size, seq_len, decoder_dim]
        values = self.value_proj(encoder_output)  # [batch_size, seq_len, decoder_dim]
        
        # Cross-attention: decoder features attend to encoder outputs
        attended_features, attention_weights = self.cross_attention(
            query=decoder_features,
            key=keys,
            value=values,
            key_padding_mask=encoder_mask
        )
        
        # Residual connection and layer norm
        decoder_features = self.norm1(decoder_features + attended_features)
        
        # Feedforward network
        ff_output = self.feedforward(decoder_features)
        decoder_features = self.norm2(decoder_features + ff_output)
        
        return decoder_features, attention_weights

class ImageDecoder(nn.Module):
    """
    Generates an image from a noise vector and text-derived context.
    Uses a transposed CNN architecture with proper cross-attention mechanism.
    """
    def __init__(self, noise_dim, text_embedding_dim, final_image_channels=3):
        super(ImageDecoder, self).__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.noise_dim = noise_dim
        
        # Initial fully connected layer
        self.fc = nn.Linear(noise_dim, 512 * 3 * 3)
        
        # CNN decoder layers with attention at multiple scales
        self.decoder_layers = nn.ModuleList([
            # Layer 1: 512 -> 256 channels, 3x3 -> 6x6
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            # Layer 2: 256 -> 128 channels, 6x6 -> 12x12  
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # Layer 3: 128 -> 64 channels, 12x12 -> 24x24
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            # Layer 4: 64 -> 32 channels, 24x24 -> 48x48
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            # Layer 5: 32 -> 16 channels, 48x48 -> 96x96
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        ])
        
        # Cross-attention layers for different CNN scales
        self.attention_layers = nn.ModuleList([
            CrossAttentionLayer(256, text_embedding_dim, num_heads=8),
            CrossAttentionLayer(128, text_embedding_dim, num_heads=8),
            CrossAttentionLayer(64, text_embedding_dim, num_heads=4),
            CrossAttentionLayer(32, text_embedding_dim, num_heads=4),
            CrossAttentionLayer(16, text_embedding_dim, num_heads=2)
        ])
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(16, final_image_channels, kernel_size=7, stride=2, padding=2),
            nn.Upsample(size=(215, 215), mode='bilinear', align_corners=False),
            nn.Tanh()
        )
        
        # Store attention weights for visualization
        self.attention_weights = []

    def forward(self, noise, encoder_output):
        """
        Forward pass for the ImageDecoder with cross-attention.

        Args:
            noise (torch.Tensor): A random noise vector [batch_size, noise_dim]
            encoder_output (torch.Tensor): The output from the TextEncoder [batch_size, seq_len, text_embedding_dim]

        Returns:
            torch.Tensor: The generated image [batch_size, 3, 215, 215]
        """
        batch_size = noise.size(0)
        
        # Start with noise through fully connected layer
        x = self.fc(noise)
        x = x.view(batch_size, 512, 3, 3)
        
        # Clear previous attention weights
        self.attention_weights = []
        
        # Pass through decoder layers with cross-attention
        for i, (decoder_layer, attention_layer) in enumerate(zip(self.decoder_layers, self.attention_layers)):
            # Apply CNN decoder layer
            x = decoder_layer(x)
            
            # Flatten spatial dimensions for attention
            b, c, h, w = x.shape
            x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # [batch_size, spatial_dim, channels]
            
            # Apply cross-attention
            attended_features, attention_weights = attention_layer(x_flat, encoder_output)
            
            # Store attention weights for visualization
            self.attention_weights.append(attention_weights.detach())
            
            # Reshape back to spatial format
            x = attended_features.permute(0, 2, 1).view(b, c, h, w)
        
        # Generate final image
        output = self.final_layer(x)
        
        return output
    
    def get_attention_weights(self):
        """
        Returns the attention weights from the last forward pass.
        Useful for visualization and analysis.
        """
        return self.attention_weights