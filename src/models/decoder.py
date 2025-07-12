import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Attention mechanism to weigh encoder outputs.
    """
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_att = nn.Linear(decoder_dim, decoder_dim)
        self.full_att = nn.Linear(decoder_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context_vector, alpha

class ImageDecoder(nn.Module):
    """
    Generates an image from a noise vector and a text-derived context vector.
    Uses a transposed CNN architecture.
    """
    def __init__(self, noise_dim, text_embedding_dim, final_image_channels=3):
        super(ImageDecoder, self).__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.noise_dim = noise_dim
        decoder_input_dim = noise_dim + text_embedding_dim

        self.attention = Attention(text_embedding_dim, text_embedding_dim)

        self.fc = nn.Linear(decoder_input_dim, 512 * 3 * 3)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # State: 256 x 6 x 6
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # State: 128 x 12 x 12
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # State: 64 x 24 x 24
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # State: 32 x 48 x 48
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # State: 16 x 96 x 96
            nn.ConvTranspose2d(16, final_image_channels, kernel_size=7, stride=2, padding=2),
            # Output: 3 x 193 x 193
            nn.Upsample(size=(215, 215), mode='bilinear', align_corners=False),
            nn.Tanh()
        )

    def forward(self, noise, encoder_output):
        """
        Forward pass for the ImageDecoder.

        Args:
            noise (torch.Tensor): A random noise vector.
            encoder_output (torch.Tensor): The output from the TextEncoder.

        Returns:
            torch.Tensor: The generated image.
        """
        # The "initial" decoder hidden state can be the mean of the encoder output
        decoder_hidden = encoder_output.mean(dim=1)

        context_vector, _ = self.attention(encoder_output, decoder_hidden)
        
        # Concatenate noise and the context vector
        decoder_input = torch.cat((noise, context_vector), dim=1)
        
        x = self.fc(decoder_input)
        x = x.view(-1, 512, 3, 3) # Reshape to a 3x3 image with 512 channels
        
        # Generate the image
        output = self.decoder(x)
        
        return output