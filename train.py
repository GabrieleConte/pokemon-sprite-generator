#!/usr/bin/env python3
"""
Training script for Pokemon Sprite Generator.

This script trains a transformer-based text encoder with CNN decoder
to generate Pokemon sprites from text descriptions.
"""

import argparse
import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models import PokemonSpriteGenerator
from src.data import create_data_loaders
from src.utils import (
    load_config, set_seed, get_device, Logger, CheckpointManager,
    print_model_summary, create_directories, save_sample_outputs,
    format_time, save_image_grid
)


def train_epoch(model, train_loader, optimizer, device, logger, epoch, config):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_attention_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['images'].to(device)
        descriptions = batch['descriptions']
        
        # Tokenize descriptions
        input_ids, attention_mask = model.text_encoder.tokenize(descriptions)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        generated_images, attention_info = model(
            input_ids, attention_mask, return_attention=True
        )
        
        # Compute loss
        total_loss_batch, loss_dict = model.compute_loss(
            generated_images, images, attention_info['attention_weights'],
            reconstruction_weight=config['training']['reconstruction_weight'],
            attention_weight=config['training']['attention_weight']
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        total_recon_loss += loss_dict['reconstruction_loss'].item()
        total_attention_loss += loss_dict['attention_loss'].item()
        num_batches += 1
        
        # Log progress
        if batch_idx % config['training']['log_every'] == 0:
            elapsed = time.time() - start_time
            logger.log(
                f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                f"Loss: {total_loss_batch.item():.6f}, "
                f"Recon: {loss_dict['reconstruction_loss'].item():.6f}, "
                f"Attention: {loss_dict['attention_loss'].item():.6f}, "
                f"Time: {format_time(elapsed)}"
            )
    
    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_attention_loss = total_attention_loss / num_batches
    
    return {
        'total_loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'attention_loss': avg_attention_loss
    }


def validate(model, val_loader, device, epoch, config):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_attention_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            images = batch['images'].to(device)
            descriptions = batch['descriptions']
            
            # Tokenize descriptions
            input_ids, attention_mask = model.text_encoder.tokenize(descriptions)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            generated_images, attention_info = model(
                input_ids, attention_mask, return_attention=True
            )
            
            # Compute loss
            total_loss_batch, loss_dict = model.compute_loss(
                generated_images, images, attention_info['attention_weights'],
                reconstruction_weight=config['training']['reconstruction_weight'],
                attention_weight=config['training']['attention_weight']
            )
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_attention_loss += loss_dict['attention_loss'].item()
            num_batches += 1
    
    # Calculate average losses
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_attention_loss = total_attention_loss / num_batches
    else:
        avg_loss = 0.0
        avg_recon_loss = 0.0
        avg_attention_loss = 0.0
    
    return {
        'total_loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'attention_loss': avg_attention_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Train Pokemon Sprite Generator')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    create_directories(config)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = Logger(config['training']['log_dir'])
    logger.log(f"Starting training with config: {args.config}")
    logger.log(f"Device: {device}")
    
    # Create data loaders
    logger.log("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=config['data']['data_dir'],
            batch_size=config['training']['batch_size'],
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split'],
            test_split=config['data']['test_split']
        )
    except:
        pass
    
    # logger.log(f"Training samples: {len(train_loader.dataset)}")
    # logger.log(f"Validation samples: {len(val_loader.dataset)}")
    # logger.log(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.log("Creating model...")
    model = PokemonSpriteGenerator(
        model_name=config['model']['text_encoder']['model_name'],
        text_embedding_dim=config['model']['decoder']['text_embedding_dim'],
        noise_dim=config['model']['decoder']['noise_dim'],
        nhead=config['model']['text_encoder']['nhead'],
        num_encoder_layers=config['model']['text_encoder']['num_encoder_layers'],
    )
    
    model = model.to(device)
    
    # Print model summary
    print_model_summary(model)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay']
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(config['training']['checkpoint_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.log(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = checkpoint_manager.load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    # Create tensorboard writer
    writer = SummaryWriter(config['training']['log_dir'])
    
    # Sample texts for visualization
    sample_texts = [
        "A small yellow electric mouse Pokemon with red cheeks and a lightning bolt tail",
        "A large orange dragon Pokemon with blue wings and a flaming tail",
        "A blue turtle Pokemon with water cannons on its shell",
        "A green plant Pokemon with a large flower on its back"
    ]
    
    # Training loop
    logger.log("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.log(f"Epoch {epoch}/{config['training']['num_epochs']} starting...")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, logger, epoch, config
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch, config)
        
        # Log metrics
        logger.log_metrics(epoch, {
            'train_loss': train_metrics['total_loss'],
            'train_recon': train_metrics['reconstruction_loss'],
            'train_attention': train_metrics['attention_loss'],
            'val_loss': val_metrics['total_loss'],
            'val_recon': val_metrics['reconstruction_loss'],
            'val_attention': val_metrics['attention_loss']
        })
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Validation', val_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Train_Reconstruction', train_metrics['reconstruction_loss'], epoch)
        writer.add_scalar('Loss/Validation_Reconstruction', val_metrics['reconstruction_loss'], epoch)
        writer.add_scalar('Loss/Train_Attention', train_metrics['attention_loss'], epoch)
        writer.add_scalar('Loss/Validation_Attention', val_metrics['attention_loss'], epoch)
        
        # Save checkpoint
        if epoch % config['training']['save_every'] == 0:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, val_metrics
            )
            logger.log(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, val_metrics, 'best_model.pth'
            )
            logger.log(f"Best model saved: {checkpoint_path}")
        
        # Generate sample outputs
        if epoch % config['training']['save_every'] == 0:
            save_sample_outputs(
                model, None, sample_texts, device, 
                'outputs/generated', epoch
            )
    
    logger.log("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()