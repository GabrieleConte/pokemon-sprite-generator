"""
Pokemon Sprite Generator

A deep learning model that generates Pokemon sprites from textual descriptions
using a Transformer-based encoder and CNN decoder with attention mechanism.
"""

__version__ = "1.0.0"
__author__ = "Pokemon Sprite Generator Team"
__email__ = "contact@pokemon-sprite-generator.com"

from .models import PokemonSpriteGenerator

__all__ = [
    'PokemonSpriteGenerator',
]