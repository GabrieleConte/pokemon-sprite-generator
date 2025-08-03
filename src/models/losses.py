"""
Perceptual loss module for Pokemon sprite generation using VGG features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from typing import List, Tuple


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Computes L1 loss between VGG features of generated and target images.
    Optimized for speed with fewer layers and efficient computation.
    """
    
    def __init__(self, feature_layers: List[int] = [8, 15], 
                 weights: List[float] = [1.0, 1.0]):
        """
        Initialize VGG perceptual loss.
        
        Args:
            feature_layers: List of VGG layer indices to use for feature extraction
            weights: List of weights for each feature layer
        """
        super().__init__()
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Load pre-trained VGG16
        vgg = vgg16(pretrained=True)
        self.vgg_features = vgg.features
        
        # Freeze VGG parameters
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from VGG at specified layers.
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            List of feature tensors
        """
        # Normalize input for VGG
        mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean_tensor) / std_tensor
        
        features = []
        for i, layer in enumerate(self.vgg_features.children()):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between generated and target images.
        
        Args:
            generated: Generated images [batch_size, 3, height, width]
            target: Target images [batch_size, 3, height, width]
            
        Returns:
            Perceptual loss scalar
        """
        # Ensure images are in [0, 1] range
        generated = torch.clamp(generated, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Skip resizing if images are close to VGG input size - major speedup!
        if generated.shape[-1] < 200:  # Only resize if significantly smaller
            generated = F.interpolate(generated, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        gen_features = self.extract_features(generated)
        target_features = self.extract_features(target)
        
        # Compute weighted feature loss
        loss = torch.tensor(0.0, device=generated.device)
        for gen_feat, target_feat, weight in zip(gen_features, target_features, self.weights):
            loss += weight * F.l1_loss(gen_feat, target_feat)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with reconstruction, perceptual, and KL divergence losses.
    """
    
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 kl_weight: float = 0.01):
        """
        Initialize combined loss.
        
        Args:
            reconstruction_weight: Weight for L1 reconstruction loss
            perceptual_weight: Weight for perceptual loss
            kl_weight: Weight for KL divergence loss
        """
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        
    def forward(self, generated: torch.Tensor, target: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            generated: Generated images [batch_size, 3, height, width]
            target: Target images [batch_size, 3, height, width]
            mu: Mean of latent distribution [batch_size, latent_dim, height, width]
            logvar: Log variance of latent distribution [batch_size, latent_dim, height, width]
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual loss components
        """
        # Denormalize images from [-1, 1] to [0, 1] for perceptual loss
        generated_norm = (generated + 1.0) / 2.0
        target_norm = (target + 1.0) / 2.0
        
        # Reconstruction loss (L1)
        recon_loss = self.l1_loss(generated, target)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(generated_norm, target_norm)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.numel())  # Normalize by number of elements
        
        # Total loss
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.kl_weight * kl_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, loss_dict


def test_perceptual_loss():
    """Test function for perceptual loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    loss_fn = VGGPerceptualLoss().to(device)
    
    # Create test inputs
    batch_size = 4
    generated = torch.randn(batch_size, 3, 215, 215).to(device)
    target = torch.randn(batch_size, 3, 215, 215).to(device)
    
    # Compute loss
    with torch.no_grad():
        loss = loss_fn(generated, target)
    
    print(f"Input shape: {generated.shape}")
    print(f"Perceptual loss: {loss.item():.4f}")
    print("Perceptual loss test passed!")


def test_combined_loss():
    """Test function for combined loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    loss_fn = CombinedLoss().to(device)
    
    # Create test inputs
    batch_size = 4
    generated = torch.randn(batch_size, 3, 215, 215).to(device)
    target = torch.randn(batch_size, 3, 215, 215).to(device)
    mu = torch.randn(batch_size, 512, 3, 3).to(device)
    logvar = torch.randn(batch_size, 512, 3, 3).to(device)
    
    # Compute loss
    with torch.no_grad():
        total_loss, loss_dict = loss_fn(generated, target, mu, logvar)
    
    print(f"Input shape: {generated.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    print("Combined loss test passed!")


if __name__ == "__main__":
    test_perceptual_loss()
    test_combined_loss()
