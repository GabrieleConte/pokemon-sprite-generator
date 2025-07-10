"""
Pokemon Sprite Generator

A deep learning model that generates Pokemon sprites from textual descriptions
using a Transformer-based encoder and CNN decoder with attention mechanism.
"""

__version__ = "1.0.0"
__author__ = "Pokemon Sprite Generator Team"
__email__ = "contact@pokemon-sprite-generator.com"

from .models import PokemonSpriteGenerator
from .data import PokemonDataset, create_data_loaders
from .utils import load_config, get_device, tensor_to_image

__all__ = [
    'PokemonSpriteGenerator',
    'PokemonDataset',
    'create_data_loaders',
    'load_config',
    'get_device',
    'tensor_to_image'
]