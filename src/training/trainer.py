import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.models.pokemon_generator import PokemonSpriteGenerator
from src.data.dataset import create_data_loaders
from src.utils.tokenization_utils import get_attention_tokens_mapping
from src.utils import get_device

class PokemonTrainer:
    """
    Comprehensive trainer for Pokemon Sprite Generator with advanced features.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 use_wandb: bool = False,
                 experiment_name: str = "pokemon_generator"):
        """
        Initialize the Pokemon trainer.
        
        Args:
            config: Training configuration dictionary
            use_wandb: Whether to use Weights & Biases for logging
            experiment_name: Name for the experiment
        """
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.experiment_name = experiment_name
        
        # Setup device
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Setup model
        self.setup_model()
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Setup optimization
        self.setup_optimization()
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def setup_directories(self):
        """Create necessary directories for saving results."""
        self.experiment_dir = Path(self.config['experiment_dir']) / self.experiment_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        self.sample_dir = self.experiment_dir / "samples"
        self.attention_dir = self.experiment_dir / "attention_maps"
        
        for dir_path in [self.experiment_dir, self.checkpoint_dir, self.log_dir, 
                        self.sample_dir, self.attention_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self):
        """Initialize the Pokemon generator model."""
        model_config = self.config['model']
        
        self.generator = PokemonSpriteGenerator(
            model_name=model_config['bert_model'],
            text_embedding_dim=model_config['text_embedding_dim'],
            noise_dim=model_config['noise_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers']
        ).to(self.device)
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.generator.parameters())} parameters")
        
    def setup_data_loaders(self):
        """Setup data loaders for training, validation, and testing."""
        data_config = self.config['data']
        
        train_loader, val_loader, test_loader = create_data_loaders(
            csv_path=data_config['csv_path'],
            image_dir=data_config['image_dir'],
            batch_size=data_config['batch_size'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            image_size=data_config['image_size'],
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        self.data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        self.logger.info(f"Data loaders created - Train: {len(train_loader)} batches, "
                        f"Val: {len(val_loader)} batches")
        
    def setup_optimization(self):
        """Setup optimizer, loss function, and learning rate scheduler."""
        opt_config = self.config['optimization']
        
        # Optimizer
        if opt_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.generator.parameters(),
                lr=opt_config['learning_rate'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.generator.parameters(),
                lr=opt_config['learning_rate'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['optimizer']}")
        
        # Loss function
        self.criterion = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Learning rate scheduler
        if opt_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=opt_config['max_epochs'],
                eta_min=opt_config['learning_rate'] * 0.01
            )
        elif opt_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=opt_config['step_size'],
                gamma=opt_config['gamma']
            )
        elif opt_config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=opt_config['patience']
            )
        else:
            self.scheduler = None
    
    def setup_monitoring(self):
        """Setup monitoring and logging tools."""
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
        
        # Weights & Biases
        if self.use_wandb:
            import wandb
            wandb.init(
                project="pokemon-sprite-generator",
                name=self.experiment_name,
                config=self.config
            )
            wandb.watch(self.generator, log_freq=100)
        elif not WANDB_AVAILABLE and self.use_wandb:
            self.logger.warning("Weights & Biases not available. Install with: pip install wandb")
    
    def compute_loss(self, generated_images: torch.Tensor, target_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined loss for training."""
        # L1 loss for sharpness
        l1_loss = self.criterion(generated_images, target_images)
        
        # MSE loss for overall structure
        mse_loss = self.mse_loss(generated_images, target_images)
        
        # Perceptual loss weight
        perceptual_weight = self.config['optimization'].get('perceptual_weight', 0)
        
        # Combined loss
        total_loss = l1_loss + perceptual_weight * mse_loss
        
        return total_loss, l1_loss, mse_loss
    
    def denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images from [-1, 1] to [0, 1]."""
        return (images + 1.0) / 2.0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        
        total_loss = 0.0
        total_l1_loss = 0.0
        total_mse_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        for batch_idx, batch in enumerate(self.data_loaders['train']):
            # Move data to device
            images = batch['image'].to(self.device)
            descriptions = batch['full_description']
            
            # Forward pass
            self.optimizer.zero_grad()
            generated_images = self.generator(descriptions)
            
            # Compute loss
            loss, l1_loss, mse_loss = self.compute_loss(generated_images, images)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                         self.config['optimization']['grad_clip'])
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_mse_loss += mse_loss.item()
            
            # Log progress
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.4f}, L1: {l1_loss.item():.4f}, MSE: {mse_loss.item():.4f}'
                )
                
                # TensorBoard logging
                self.tb_writer.add_scalar('Loss/Train_Batch', loss.item(), self.global_step)
                self.tb_writer.add_scalar('Loss/L1_Batch', l1_loss.item(), self.global_step)
                self.tb_writer.add_scalar('Loss/MSE_Batch', mse_loss.item(), self.global_step)
                
                # Weights & Biases logging
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_l1_loss': l1_loss.item(),
                        'batch_mse_loss': mse_loss.item(),
                        'epoch': epoch,
                        'global_step': self.global_step
                    })
            
            self.global_step += 1
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        
        return {
            'loss': avg_loss,
            'l1_loss': avg_l1_loss,
            'mse_loss': avg_mse_loss
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.generator.eval()
        
        total_loss = 0.0
        total_l1_loss = 0.0
        total_mse_loss = 0.0
        num_batches = len(self.data_loaders['val'])
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loaders['val']):
                # Move data to device
                images = batch['image'].to(self.device)
                descriptions = batch['full_description']
                
                # Forward pass
                generated_images = self.generator(descriptions)
                
                # Compute loss
                loss, l1_loss, mse_loss = self.compute_loss(generated_images, images)
                
                # Update metrics
                total_loss += loss.item()
                total_l1_loss += l1_loss.item()
                total_mse_loss += mse_loss.item()
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        
        return {
            'loss': avg_loss,
            'l1_loss': avg_l1_loss,
            'mse_loss': avg_mse_loss
        }
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate sample images for monitoring."""
        self.generator.eval()
        
        # Get a batch of validation data
        val_batch = next(iter(self.data_loaders['val']))
        descriptions = val_batch['full_description'][:num_samples]
        real_images = val_batch['image'][:num_samples]
        
        with torch.no_grad():
            generated_images = self.generator(descriptions)
            attention_weights = self.generator.get_attention_weights()
        
        # Convert to numpy and denormalize
        real_images = self.denormalize_images(real_images)
        generated_images = self.denormalize_images(generated_images)
        
        # Create comparison grid
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        for i in range(num_samples):
            # Real image
            axes[0, i].imshow(real_images[i].permute(1, 2, 0).cpu())
            axes[0, i].set_title(f"Real {val_batch['name'][i]}")
            axes[0, i].axis('off')
            
            # Generated image
            axes[1, i].imshow(generated_images[i].permute(1, 2, 0).cpu())
            axes[1, i].set_title(f"Generated {val_batch['name'][i]}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.sample_dir / f'epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save attention visualization for first sample
        if attention_weights:
            self.visualize_attention(descriptions[0], attention_weights, epoch)
        
        # Log to tensorboard
        self.tb_writer.add_images('Real_Images', real_images, epoch)
        self.tb_writer.add_images('Generated_Images', generated_images, epoch)
        
        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                'real_images': [wandb.Image(img) for img in real_images],
                'generated_images': [wandb.Image(img) for img in generated_images],
                'epoch': epoch
            })
    
    def visualize_attention(self, description: str, attention_weights: List[torch.Tensor], epoch: int):
        """Visualize attention weights."""
        try:
            # Get proper tokens
            tokens = get_attention_tokens_mapping([description], attention_weights)
            
            # Create attention visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (weights, ax) in enumerate(zip(attention_weights[:6], axes)):
                # Average attention over spatial dimensions
                avg_attention = torch.mean(weights[0], dim=0).cpu().numpy()
                
                # Create bar plot
                ax.bar(range(len(avg_attention)), avg_attention)
                ax.set_title(f'Layer {i+1} Attention')
                ax.set_xlabel('Token Index')
                ax.set_ylabel('Attention Weight')
                
                # Add token labels (first 10 tokens)
                if len(tokens[0]) > 0:
                    tick_labels = tokens[0][:min(10, len(avg_attention))]
                    ax.set_xticks(range(len(tick_labels)))
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.attention_dir / f'attention_epoch_{epoch:04d}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not create attention visualization: {e}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        start_epoch = self.current_epoch
        max_epochs = self.config['training']['max_epochs']
        
        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Time: {epoch_time:.2f}s"
            )
            
            # TensorBoard logging
            self.tb_writer.add_scalar('Loss/Train_Epoch', train_metrics['loss'], epoch)
            self.tb_writer.add_scalar('Loss/Val_Epoch', val_metrics['loss'], epoch)
            self.tb_writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Weights & Biases logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })
            
            # Generate samples
            if epoch % self.config['logging']['sample_interval'] == 0:
                self.generate_samples(epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if epoch % self.config['logging']['checkpoint_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.logger.info("Training completed!")
        
        # Final evaluation
        self.evaluate()
    
    def evaluate(self):
        """Evaluate the model on test set."""
        self.logger.info("Evaluating on test set...")
        
        self.generator.eval()
        total_loss = 0.0
        num_batches = len(self.data_loaders['test'])
        
        with torch.no_grad():
            for batch in self.data_loaders['test']:
                images = batch['image'].to(self.device)
                descriptions = batch['full_description']
                
                generated_images = self.generator(descriptions)
                loss, _, _ = self.compute_loss(generated_images, images)
                total_loss += loss.item()
        
        avg_test_loss = total_loss / num_batches
        self.logger.info(f"Test Loss: {avg_test_loss:.4f}")
        
        if self.use_wandb:
            import wandb
            wandb.log({'test_loss': avg_test_loss})
        
        return avg_test_loss

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

if __name__ == "__main__":
    # Load configuration
    config_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    config = load_config(config_path)
    
    # Create trainer
    trainer = PokemonTrainer(config, use_wandb=False)
    
    # Start training
    trainer.train()
