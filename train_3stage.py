#!/usr/bin/env python3
"""
Three-stage training script for Pokemon sprite generation with Stable Diffusion-like architecture.

Stage 1: Train VAE with perceptual loss
Stage 2: Train U-Net for diffusion denoising
Stage 3: Fine-tune text encoder with frozen VAE and U-Net
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.vae_trainer import VAETrainer
from src.training.diffusion_trainer import DiffusionTrainer
from src.training.final_trainer import FinalTrainer
from src.training.vae_trainer import load_config
from src.data import get_dataset_statistics


def main():
    parser = argparse.ArgumentParser(description='Train Pokemon Generator (3-Stage Training)')
    parser.add_argument('--config', 
                       default='config/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--stage', 
                       choices=['1', '2', '3', 'all'],
                       default='all',
                       help='Training stage to run (1: VAE, 2: Diffusion, 3: Final, all: All stages)')
    parser.add_argument('--vae-checkpoint', 
                       default=None,
                       help='Path to pre-trained VAE checkpoint (required for stages 2 and 3)')
    parser.add_argument('--diffusion-checkpoint', 
                       default=None,
                       help='Path to pre-trained diffusion checkpoint (required for stage 3)')
    parser.add_argument('--experiment-name', 
                       default='pokemon_3stage',
                       help='Name for the experiment')
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
    
    # Create experiment directory
    experiment_dir = Path(config['experiment_dir'])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: VAE Training
    if args.stage in ['1', 'all']:
        print("="*50)
        print("STAGE 1: VAE Training with Perceptual Loss")
        print("="*50)
        
        vae_trainer = VAETrainer(
            config=config,
            experiment_name=f"{args.experiment_name}_vae"
        )
        
        if args.resume and args.stage == '1':
            print(f"Resuming VAE training from checkpoint: {args.resume}")
            vae_trainer.load_checkpoint(args.resume)
        
        vae_trainer.train()
        
        # Set VAE checkpoint path for next stage
        vae_checkpoint_path = experiment_dir / f"{args.experiment_name}_vae" / "checkpoints" / "vae_best_model.pth"
        
        print(f"Stage 1 complete! VAE checkpoint saved to: {vae_checkpoint_path}")
    
    # Stage 2: Diffusion Training
    if args.stage in ['2', 'all']:
        print("="*50)
        print("STAGE 2: U-Net Diffusion Training")
        print("="*50)
        
        # Determine VAE checkpoint path
        if args.vae_checkpoint:
            vae_checkpoint_path = args.vae_checkpoint
        elif args.stage == 'all':
            vae_checkpoint_path = experiment_dir / f"{args.experiment_name}_vae" / "checkpoints" / "vae_best_model.pth"
        else:
            raise ValueError("VAE checkpoint path required for stage 2. Use --vae-checkpoint argument.")
        
        if not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")
        
        diffusion_trainer = DiffusionTrainer(
            config=config,
            vae_checkpoint_path=vae_checkpoint_path,
            experiment_name=f"{args.experiment_name}_diffusion"
        )
        
        if args.resume and args.stage == '2':
            print(f"Resuming diffusion training from checkpoint: {args.resume}")
            diffusion_trainer.load_checkpoint(args.resume)
        
        diffusion_trainer.train()
        
        # Set diffusion checkpoint path for next stage
        diffusion_checkpoint_path = experiment_dir / f"{args.experiment_name}_diffusion" / "checkpoints" / "diffusion_best_model.pth"
        
        print(f"Stage 2 complete! Diffusion checkpoint saved to: {diffusion_checkpoint_path}")
    
    # Stage 3: Final Training (Text Encoder Fine-tuning)
    if args.stage in ['3', 'all']:
        print("="*50)
        print("STAGE 3: Final Training (Text Encoder Fine-tuning)")
        print("="*50)
        
        # Determine checkpoint paths
        if args.vae_checkpoint:
            vae_checkpoint_path = args.vae_checkpoint
        elif args.stage == 'all':
            vae_checkpoint_path = experiment_dir / f"{args.experiment_name}_vae" / "checkpoints" / "vae_best_model.pth"
        else:
            raise ValueError("VAE checkpoint path required for stage 3. Use --vae-checkpoint argument.")
        
        if args.diffusion_checkpoint:
            diffusion_checkpoint_path = args.diffusion_checkpoint
        elif args.stage == 'all':
            diffusion_checkpoint_path = experiment_dir / f"{args.experiment_name}_diffusion" / "checkpoints" / "diffusion_best_model.pth"
        else:
            raise ValueError("Diffusion checkpoint path required for stage 3. Use --diffusion-checkpoint argument.")
        
        if not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")
        
        if not os.path.exists(diffusion_checkpoint_path):
            raise FileNotFoundError(f"Diffusion checkpoint not found at {diffusion_checkpoint_path}")
        
        # Note: We'll need to create a modified final trainer that works with the U-Net
        # For now, we'll use the existing final trainer with the VAE checkpoint
        final_trainer = FinalTrainer(
            config=config,
            vae_checkpoint_path=vae_checkpoint_path,
            diffusion_checkpoint_path=diffusion_checkpoint_path,
            experiment_name=f"{args.experiment_name}_final"
        )
        
        if args.resume and args.stage == '3':
            print(f"Resuming final training from checkpoint: {args.resume}")
            final_trainer.load_checkpoint(args.resume)
        
        final_trainer.train()
        
        print(f"Stage 3 complete! Final model saved!")
    
    print("="*50)
    print("ALL STAGES COMPLETE!")
    print("="*50)
    print("Your Pokemon sprite generator is ready!")
    print(f"Experiment files saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
