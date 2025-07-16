#!/usr/bin/env python3
"""
Stage 2: Fine-tune text encoder with pre-trained VAE decoder.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.final_trainer import FinalTrainer, load_config
from src.data import get_dataset_statistics


def main():
    parser = argparse.ArgumentParser(description='Train Pokemon Generator (Stage 2)')
    parser.add_argument('--config', 
                       default='config/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--vae-checkpoint', 
                       required=True,
                       help='Path to pre-trained VAE checkpoint from stage 1')
    parser.add_argument('--experiment-name', 
                       default='pokemon_final_stage2',
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
    
    # Check if VAE checkpoint exists
    if not os.path.exists(args.vae_checkpoint):
        print(f"Error: VAE checkpoint not found at {args.vae_checkpoint}")
        print("Please run train_stage1.py first to train the VAE.")
        return
    
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
    print(f"Creating final trainer for experiment: {args.experiment_name}")
    print(f"Loading VAE from: {args.vae_checkpoint}")
    trainer = FinalTrainer(
        config=config,
        vae_checkpoint_path=args.vae_checkpoint,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting final training (Stage 2)...")
    print("This stage fine-tunes the text encoder with the pre-trained VAE decoder.")
    print("Phase 1: Train only text encoder")
    print("Phase 2: Joint training of text encoder and VAE decoder")
    trainer.train()
    
    print("Stage 2 complete! Your Pokemon sprite generator is ready!")

if __name__ == "__main__":
    main()
