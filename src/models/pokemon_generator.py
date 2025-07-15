import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .decoder import ImageDecoder

class PokemonSpriteGenerator(nn.Module):
    """
    The main Generator model that combines the TextEncoder and ImageDecoder.
    """
    def __init__(self, model_name='prajjwal1/bert-mini', text_embedding_dim=256, noise_dim=100, nhead=4, num_encoder_layers=2):
        super(PokemonSpriteGenerator, self).__init__()
        self.encoder = TextEncoder(
            model_name=model_name, 
            hidden_dim=text_embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers
        )
        self.decoder = ImageDecoder(
            noise_dim=noise_dim, 
            text_embedding_dim=text_embedding_dim
        )
        self.noise_dim = noise_dim

    def forward(self, text_list):
        """
        Forward pass for the Generator.

        Args:
            text_list (list of str): A list of text descriptions.

        Returns:
            torch.Tensor: The generated image.
        """
        # 1. Encode the text
        encoder_output = self.encoder(text_list)
        
        # 2. Generate noise
        batch_size = encoder_output.size(0)
        noise = torch.randn(batch_size, self.noise_dim).to(encoder_output.device)
        
        # 3. Generate an image
        generated_image = self.decoder(noise, encoder_output)
        
        return generated_image
    
    def get_attention_weights(self):
        """
        Returns the attention weights from the last forward pass.
        Useful for visualization and analysis.
        """
        return self.decoder.get_attention_weights()
