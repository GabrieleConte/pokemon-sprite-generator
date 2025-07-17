"""
Pokemon Sprite Generator

A deep learning model that generates Pokemon sprites from textual descriptions
using VAE and diffusion models.
"""

__version__ = "1.0.0"
__author__ = "Pokemon Sprite Generator Team"
__email__ = "contact@pokemon-sprite-generator.com"

# Import main components
from .models import TextEncoder, UNet, CombinedLoss, VGGPerceptualLoss, VAEDecoder
from .training import VAETrainer, DiffusionTrainer, FinalTrainer

__all__ = [
    'TextEncoder',
    'UNet', 
    'CombinedLoss',
    'VGGPerceptualLoss',
    'VAEDecoder',
    'VAETrainer',
    'DiffusionTrainer',
    'FinalTrainer'
]