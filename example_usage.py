#!/usr/bin/env python3
"""
Example usage script for Pokemon Sprite Generator training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.trainer import PokemonTrainer, load_config
from src.data import get_dataset_statistics, create_data_loaders

def main():
    # Example 1: Show dataset statistics
    print("Example 1: Dataset Statistics")
    print("-" * 50)
    
    config = load_config('config/train_config.yaml')
    
    try:
        stats = get_dataset_statistics(
            config['data']['csv_path'],
            config['data']['image_dir']
        )
        
        print(f"Total Pokemon: {stats['total_samples']}")
        print(f"Average description length: {stats['avg_description_length']:.1f} words")
        print(f"Description length std: {stats['description_length_std']:.1f}")
        print(f"Top 5 Pokemon types: {list(stats['type_distribution'].items())[:5]}")
        print(f"Missing images: {stats['missing_images']}")
        print(f"Missing descriptions: {stats['missing_descriptions']}")
    except Exception as e:
        print(f"Error computing statistics: {e}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 2: Test data loaders
    print("Example 2: Data Loaders Test")
    print("-" * 50)
    
    try:
        train_loader, val_loader = create_data_loaders(
            csv_path=config['data']['csv_path'],
            image_dir=config['data']['image_dir'],
            batch_size=4,
            image_size=config['data']['image_size']
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print("Data loaders created successfully")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 3: Initialize trainer (without training)
    print("Example 3: Trainer Initialization")
    print("-" * 50)
    
    try:
        trainer = PokemonTrainer(
            config=config,
            use_wandb=False,
            experiment_name='test_experiment'
        )
        
        print(f"Trainer initialized successfully")
        print(f"Model device: {trainer.device}")
        print(f"Experiment name: test_experiment")
        print(f"Using Weights & Biases: False")
        
    except Exception as e:
        print(f"Error initializing trainer: {e}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 4: Training commands
    print("Example 4: Training Commands")
    print("-" * 50)
    
    print("To start training:")
    print("  python train.py --config config/train_config.yaml")
    print("")
    print("To resume training:")
    print("  python train.py --resume checkpoints/checkpoint_epoch_10.pth")
    print("")
    print("To use Weights & Biases:")
    print("  python train.py --use-wandb --experiment-name my_experiment")
    print("")
    print("To show dataset statistics:")
    print("  python train.py --data-stats")

if __name__ == "__main__":
    main()