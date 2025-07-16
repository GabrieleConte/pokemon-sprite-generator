#!/usr/bin/env python3
"""
Stage 1: Train VAE encoder/decoder with text conditioning.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.vae_trainer import VAETrainer, load_config
from src.data import get_dataset_statistics


def main():
    parser = argparse.ArgumentParser(description='Train Pokemon VAE (Stage 1)')
    parser.add_argument('--config', 
                       default='config/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--experiment-name', 
                       default='pokemon_vae_stage1',
                       help='Name for the experiment')
    parser.add_argument('--use-wandb', 
                       action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--resume', 
                       default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data-stats', 
                       action='store_true',
                       help='Show dataset statistics and exit')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Show dataset statistics if requested
    if args.data_stats:
        print("Computing dataset statistics...")
        stats = get_dataset_statistics(
            config['data']['csv_path'],
            config['data']['image_dir']
        )
        
        print("\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Average description length: {stats['avg_description_length']:.1f} words")
        print(f"  Description length std: {stats['description_length_std']:.1f}")
        print(f"  Top Pokemon types: {list(stats['type_distribution'].items())[:10]}")
        return
    
    # Create trainer
    print(f"Creating VAE trainer for experiment: {args.experiment_name}")
    trainer = VAETrainer(
        config=config,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting VAE training (Stage 1)...")
    print("This stage trains the VAE encoder/decoder with text conditioning.")
    print("The text encoder (BERT) is frozen during this stage.")
    trainer.train()
    
    print("Stage 1 complete! You can now run train_stage2.py to fine-tune the text encoder.")

if __name__ == "__main__":
    main()
