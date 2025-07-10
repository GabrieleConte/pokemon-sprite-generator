import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import TextEncoder
from .cnn_decoder import CNNDecoder
from .attention import AttentionMechanism, CrossAttention


class PokemonSpriteGenerator(nn.Module):
    """Main model combining text encoder, attention mechanism, and CNN decoder."""
    
    def __init__(self, 
                 # Text encoder parameters
                 vocab_size=30522,
                 embedding_dim=256,
                 num_heads=8,
                 num_layers=6,
                 hidden_dim=512,
                 max_seq_length=128,
                 # Decoder parameters
                 latent_dim=256,
                 noise_dim=100,
                 image_size=215,
                 num_channels=3,
                 base_channels=64,
                 # Attention parameters
                 attention_dim=256,
                 dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.image_size = image_size
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # CNN decoder
        self.cnn_decoder = CNNDecoder(
            latent_dim=latent_dim,
            noise_dim=noise_dim,
            image_size=image_size,
            num_channels=num_channels,
            base_channels=base_channels
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(
            encoder_dim=embedding_dim,
            decoder_dim=latent_dim,
            hidden_dim=attention_dim
        )
        
        # Cross-attention for more sophisticated text-image interaction
        self.cross_attention = CrossAttention(
            text_dim=embedding_dim,
            image_dim=latent_dim,
            hidden_dim=attention_dim
        )
        
        # Text-to-latent projection
        self.text_to_latent = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, input_ids, attention_mask=None, noise=None, return_attention=False):
        """
        Forward pass of the complete model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            noise: Random noise [batch_size, noise_dim] or None
            return_attention: Whether to return attention weights
            
        Returns:
            generated_images: Generated sprites [batch_size, channels, height, width]
            attention_weights: Attention weights (if return_attention=True)
        """
        # Encode text
        encoder_outputs, pooled_output = self.text_encoder(input_ids, attention_mask)
        
        # Project pooled text features to latent space
        text_latent = self.text_to_latent(pooled_output)  # [batch_size, latent_dim]
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(
            encoder_outputs, text_latent, attention_mask
        )
        
        # Apply cross-attention for enhanced features
        enhanced_features, cross_attention_weights = self.cross_attention(
            encoder_outputs, context_vector, attention_mask
        )
        
        # Generate images using CNN decoder
        generated_images = self.cnn_decoder(enhanced_features, noise)
        
        if return_attention:
            return generated_images, {
                'attention_weights': attention_weights,
                'cross_attention_weights': cross_attention_weights
            }
        
        return generated_images
    
    def generate(self, texts, num_samples=1, temperature=1.0, return_attention=False):
        """
        Generate Pokemon sprites from text descriptions.
        
        Args:
            texts: List of text descriptions or single text
            num_samples: Number of samples to generate per text
            temperature: Sampling temperature for noise
            return_attention: Whether to return attention weights
            
        Returns:
            generated_images: Generated sprites
            attention_weights: Attention weights (if return_attention=True)
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize texts
            input_ids, attention_mask = self.text_encoder.tokenize(texts)
            input_ids = input_ids.to(next(self.parameters()).device)
            attention_mask = attention_mask.to(next(self.parameters()).device)
            
            batch_size = input_ids.shape[0]
            
            if num_samples > 1:
                # Repeat inputs for multiple samples
                input_ids = input_ids.repeat_interleave(num_samples, dim=0)
                attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)
            
            # Generate noise with temperature
            noise = torch.randn(
                batch_size * num_samples, self.noise_dim, 
                device=input_ids.device
            ) * temperature
            
            # Generate images
            if return_attention:
                generated_images, attention_info = self.forward(
                    input_ids, attention_mask, noise, return_attention=True
                )
                return generated_images, attention_info
            else:
                generated_images = self.forward(input_ids, attention_mask, noise)
                return generated_images
    
    def compute_loss(self, generated_images, target_images, attention_weights=None, 
                    reconstruction_weight=1.0, attention_weight=0.1):
        """
        Compute training loss.
        
        Args:
            generated_images: Generated sprites [batch_size, channels, height, width]
            target_images: Target sprites [batch_size, channels, height, width]
            attention_weights: Attention weights for regularization
            reconstruction_weight: Weight for reconstruction loss
            attention_weight: Weight for attention regularization
            
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss (L1 + L2)
        l1_loss = F.l1_loss(generated_images, target_images)
        l2_loss = F.mse_loss(generated_images, target_images)
        reconstruction_loss = l1_loss + l2_loss
        
        # Attention regularization (encourage diversity)
        attention_loss = 0.0
        if attention_weights is not None:
            # Encourage attention to be distributed (not too concentrated)
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
            attention_loss = -torch.mean(attention_entropy)  # Negative because we want to maximize entropy
        
        # Total loss
        total_loss = reconstruction_weight * reconstruction_loss + attention_weight * attention_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'attention_loss': attention_loss
        }
        
        return total_loss, loss_dict
    
    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path, device='cpu'):
        """Load model state dict."""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)