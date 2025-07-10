# Pokémon Sprite Generator

A deep learning model that generates Pokémon sprites from textual descriptions using a Transformer-based encoder and CNN decoder with attention mechanism.

## Features

- **Text Encoder**: Transformer-based encoder with pre-trained BERT-mini embeddings (256 dimensions)
- **CNN Decoder**: Convolutional Neural Network for sprite generation (215x215 pixels)
- **Attention Mechanism**: Allows decoder to focus on relevant words during image generation
- **End-to-End Training**: Complete training pipeline on Pokémon dataset

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py --config config/train_config.yaml
```

### Inference
```bash
python generate.py --text "A blue dragon-type Pokemon with large wings and fiery breath"
```

## Model Architecture

The model consists of:
1. **Text Encoder**: Processes input descriptions using BERT-mini embeddings and Transformer layers
2. **Attention Mechanism**: Links encoder output with decoder states for selective focus
3. **CNN Decoder**: Generates 215x215 pixel sprites using transposed convolutions

## Dataset

Uses the Pokémon dataset from: https://github.com/cristobalmitchell/pokedex/