# Pokemon Sprite Generator

A deep learning project that generates Pokemon sprites from text descriptions using a transformer-based text encoder with CNN decoder architecture.

## Features

- **Text-to-Image Generation**: Generate Pokemon sprites from natural language descriptions
- **Attention Mechanism**: Cross-attention between text and image features for better generation quality
- **Comprehensive Training Pipeline**: Full-featured training system with monitoring and visualization
- **Data Augmentation**: Robust data preprocessing and augmentation
- **Experiment Tracking**: Support for TensorBoard and Weights & Biases
- **Checkpointing**: Automatic model saving and resuming from checkpoints
- **Attention Visualization**: Tools to visualize what the model focuses on

## Architecture

The model consists of:
1. **Text Encoder**: BERT-based transformer (prajjwal1/bert-mini) for encoding Pokemon descriptions
2. **Cross-Attention**: Attention mechanism between text and image features
3. **CNN Decoder**: Convolutional decoder that generates 64x64 Pokemon sprites
4. **Loss Function**: Combination of reconstruction loss (L1 + MSE) and attention regularization

## Dataset

The model expects a Pokemon dataset with:
- **CSV file** (`data/pokemon.csv`): Contains Pokemon information with columns:
  - `national_number`: Pokemon ID (001-898)
  - `english_name`: Pokemon name
  - `description`: Text description of the Pokemon
- **Images folder** (`data/small_images/`): Contains 64x64 PNG images named `001.png` to `898.png`

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pokemon-sprite-generator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Check Dataset Statistics

```bash
python train.py --data-stats
```

### 2. Start Training

```bash
# Basic training
python train.py --config config/train_config.yaml

# With Weights & Biases logging
python train.py --use-wandb --experiment-name my_experiment

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pth
```

### 3. Monitor Training

- **TensorBoard**: View training metrics and generated samples
  ```bash
  tensorboard --logdir logs/
  ```

- **Weights & Biases**: If enabled, view experiments at https://wandb.ai

### 4. Example Usage

```bash
# Run examples to test the system
python example_usage.py
```

## Configuration

Edit `config/train_config.yaml` to customize training parameters:

```yaml
# Model configuration
model:
  text_encoder:
    model_name: "prajjwal1/bert-mini"
    hidden_dim: 256
    nhead: 8
    num_encoder_layers: 6
  
  decoder:
    noise_dim: 100
    text_embedding_dim: 256
    base_channels: 512

# Training configuration
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0002
  weight_decay: 0.0001
  
  # Loss weights
  reconstruction_weight: 1.0
  attention_weight: 0.1

# Data configuration
data:
  csv_path: "data/pokemon.csv"
  image_dir: "data/small_images"
  image_size: 64
  train_split: 0.8
  val_split: 0.2
```

## Project Structure

```
pokemon-sprite-generator/
├── config/
│   └── train_config.yaml          # Training configuration
├── data/
│   ├── pokemon.csv                # Pokemon dataset
│   └── small_images/              # 64x64 Pokemon images
├── src/
│   ├── data/
│   │   └── dataset.py             # Dataset and data loading
│   ├── models/
│   │   └── pokemon_generator.py   # Model architecture
│   ├── training/
│   │   └── trainer.py             # Training pipeline
│   └── utils/
│       ├── attention_utils.py     # Attention visualization
│       └── tokenization_utils.py  # Text tokenization
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
├── outputs/                       # Generated samples
├── train.py                       # Main training script
├── example_usage.py               # Usage examples
└── requirements.txt               # Dependencies
```

## Training Pipeline Features

### PokemonTrainer Class

The `PokemonTrainer` class provides:

- **Automatic Data Loading**: Creates train/validation splits from your dataset
- **Model Initialization**: Sets up the Pokemon generator model with proper configuration
- **Loss Computation**: Combines reconstruction and attention losses
- **Monitoring**: Tracks training metrics and generates sample images
- **Checkpointing**: Saves model state for resuming training
- **Visualization**: Creates attention maps and training progress plots

### Key Features

1. **Robust Data Loading**: Handles missing images/descriptions gracefully
2. **Mixed Loss Functions**: L1 + MSE reconstruction loss with attention regularization
3. **Sample Generation**: Automatically generates samples during training for monitoring
4. **Attention Visualization**: Creates attention maps to understand model focus
5. **Experiment Tracking**: Supports both TensorBoard and Weights & Biases
6. **Gradient Clipping**: Prevents gradient explosion during training
7. **Learning Rate Scheduling**: Automatically adjusts learning rate during training

## Attention Mechanism

The model uses cross-attention between text and image features:

- **Text Tokens**: BERT tokenization of Pokemon descriptions
- **Image Features**: CNN feature maps from intermediate decoder layers
- **Attention Weights**: Learnable attention between text tokens and spatial locations
- **Visualization**: Tools to visualize which words the model focuses on for each image region

## Monitoring and Visualization

### TensorBoard Metrics

- Training/validation losses
- Reconstruction and attention loss components
- Generated sample images
- Attention weight visualizations
- Learning rate schedules

### Weights & Biases Integration

- Experiment tracking across multiple runs
- Hyperparameter sweeps
- Model artifact storage
- Collaborative experiment sharing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config
2. **Missing Images**: Check image paths and file permissions
3. **Tokenization Issues**: Ensure proper text preprocessing
4. **Attention Errors**: Verify sequence length matching between text and attention

### Performance Tips

1. **GPU Memory**: Use smaller batch sizes for limited GPU memory
2. **Training Speed**: Increase `num_workers` in data loading
3. **Generation Quality**: Adjust loss weights in configuration
4. **Monitoring**: Use `--data-stats` to verify dataset before training

## Development

### Adding New Features

1. **New Loss Functions**: Add to `trainer.py` in the loss computation
2. **Data Augmentation**: Modify dataset transforms in `dataset.py`
3. **Architecture Changes**: Update model in `models/pokemon_generator.py`
4. **Visualization Tools**: Add to `utils/` directory

### Testing

```bash
# Test dataset loading
python -c "from src.data.dataset import get_dataset_statistics; print(get_dataset_statistics('data/pokemon.csv', 'data/small_images'))"

# Test model creation
python -c "from src.training.trainer import PokemonTrainer, load_config; trainer = PokemonTrainer(load_config('config/train_config.yaml'))"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BERT model: prajjwal1/bert-mini from Hugging Face
- Pokemon dataset: Original Pokemon descriptions and sprites
- PyTorch: Deep learning framework
- Weights & Biases: Experiment tracking platform