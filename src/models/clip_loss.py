"""
CLIP loss for text-image alignment in Pokemon sprite generation using OpenAI CLIP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformers import CLIPModel, CLIPProcessor


class CLIPLoss(nn.Module):
    """
    CLIP loss using pre-trained OpenAI CLIP model for text-image alignment.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load the pre-trained CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move model to device
        if isinstance(device, str):
            self.model = self.model.to(device)
        else:
            self.model = self.model.to(device)
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, text):
        """
        Compute CLIP loss between images and text.
        
        Args:
            image: (B, 3, H, W), range [-1, 1]
            text: list of strings (length B)
            
        Returns:
            loss: CLIP alignment loss
        """
        # Clean text by removing special tokens
        clean_text = []
        for t in text:
            clean_text.append(' '.join([word for word in t.split(' ') if word not in ['[MASK]', '[NAME]']]))

        # Convert image from [-1, 1] to [0, 1]
        image = (image + 1) / 2

        # Process inputs
        inputs = self.processor(text=clean_text, images=image, return_tensors="pt", padding=True, truncation=True,
                                do_rescale=False).to(self.device)

        # Get CLIP outputs
        outputs = self.model(**inputs)
        image_features = F.normalize(outputs.image_embeds, dim=-1)
        text_features = F.normalize(outputs.text_embeds, dim=-1)

        # Compute cosine similarity
        similarity = torch.sum(image_features * text_features, dim=-1)

        # Return negative similarity as loss (we want to maximize similarity)
        loss = -similarity.mean()

        return loss


def test_clip_loss():
    """Test function for CLIP loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    clip_model = CLIPLoss(device=str(device)).to(device)
    
    # Create test inputs - images in [-1, 1] range as expected by the model
    batch_size = 4
    images = torch.randn(batch_size, 3, 215, 215).clamp(-1, 1).to(device)
    text_list = [
        "A small green creature with a bulb on its back",
        "A fire-breathing orange dragon with wings", 
        "A blue turtle with a hard shell",
        "A yellow electric mouse Pokemon"
    ]
    
    # Forward pass
    clip_loss = clip_model(images, text_list)
    
    print(f"Images shape: {images.shape}")
    print(f"Text list length: {len(text_list)}")
    print(f"CLIP loss: {clip_loss.item():.4f}")
    print(f"Model parameters: {sum(p.numel() for p in clip_model.parameters()):,}")
    
    assert clip_loss.numel() == 1, "Loss should be scalar"
    print("CLIP loss test passed!")


if __name__ == "__main__":
    test_clip_loss()
