import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import yaml

from src.models.vae_decoder import PokemonVAE
from src.models.text_encoder import TextEncoder
from src.models.losses import CombinedLoss
from src.data import create_data_loaders
from src.utils import get_device


class VAETrainer:
    """
    Trainer for the VAE stage - learns to encode/decode Pokemon images with text conditioning.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 experiment_name: str = "pokemon_vae"):
        """
        Initialize the VAE trainer.
        
        Args:
            config: Training configuration dictionary
            experiment_name: Name for the experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # Setup device
        self.device = get_device()
        
        # KL annealing parameters
        self.kl_anneal_start = config['training'].get('kl_anneal_start', 0)
        self.kl_anneal_end = config['training'].get('kl_anneal_end', 50)
        self.kl_weight_start = config['training'].get('kl_weight_start', 0.0)
        self.kl_weight_end = config['training'].get('kl_weight_end', 1.0)
        self.free_bits = config['training'].get('free_bits', 0.5)  # Minimum KL per dimension
        self.kl_annealing = config['training'].get('kl_annealing', True)  # Enable KL annealing by default
        
        # Current training state
        self.current_epoch = 0
        self.global_step = 0
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
        
        for dir_path in [self.experiment_dir, self.checkpoint_dir, self.log_dir, self.sample_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'vae_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self):
        """Initialize the VAE model and text encoder."""
        model_config = self.config['model']
        
        # Text encoder (frozen during VAE training)
        self.text_encoder = TextEncoder(
            model_name=model_config['bert_model'],
            hidden_dim=model_config['text_embedding_dim']
        ).to(self.device)
        
        # Freeze text encoder for VAE training
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # VAE model
        self.vae = PokemonVAE(
            latent_dim=model_config.get('latent_dim', 1024),
            text_dim=model_config['text_embedding_dim']
        ).to(self.device)
        
        self.logger.info(f"VAE initialized with {sum(p.numel() for p in self.vae.parameters())} parameters")
        
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
        """Setup optimizer and learning rate scheduler."""
        opt_config = self.config['optimization']
        
        # Optimizer for VAE
        if opt_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.vae.parameters(),
                lr=opt_config['learning_rate'],
                betas=(opt_config['beta1'], opt_config['beta2'])
            )
        elif opt_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.vae.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        
        # Learning rate scheduler
        if opt_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['vae_epochs']
            )
        elif opt_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=opt_config['step_size'],
                gamma=opt_config['gamma']
            )
        
        # Loss function with perceptual loss
        self.criterion = CombinedLoss(
            reconstruction_weight=self.config['training'].get('reconstruction_weight', 1.0),
            perceptual_weight=self.config['training'].get('perceptual_weight', 0.1),
            kl_weight=self.config['training'].get('kl_weight', 0.01)
        ).to(self.device)
        
    def setup_monitoring(self):
        """Setup monitoring and logging tools."""
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
    
    def get_kl_weight(self, epoch: int) -> float:
        """Get current KL weight with annealing."""
        if epoch < self.kl_anneal_start:
            return self.kl_weight_start
        elif epoch >= self.kl_anneal_end:
            return self.kl_weight_end
        else:
            # Linear annealing
            progress = (epoch - self.kl_anneal_start) / (self.kl_anneal_end - self.kl_anneal_start)
            return self.kl_weight_start + progress * (self.kl_weight_end - self.kl_weight_start)
    
    def compute_free_bits_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL loss with free bits to prevent posterior collapse."""
        # Standard KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply free bits: max(KL_per_dim, free_bits)
        kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        
        # Sum over latent dimensions, mean over batch
        kl_loss = kl_per_dim.sum(dim=-1).mean()
        
        return kl_loss
    
    def compute_vae_loss(self, outputs: Dict[str, torch.Tensor], target_images: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VAE loss with perceptual loss, KL annealing and free bits."""
        reconstructed = outputs['reconstructed']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Use combined loss (perceptual + KL)
        total_loss, loss_dict = self.criterion(reconstructed, target_images, mu, logvar)
        
        # Apply KL annealing
        kl_weight = self.get_kl_weight(epoch)
        
        # Recompute total loss with annealed KL weight by recalculating from tensors
        if self.kl_annealing:
            # Get the individual loss components as tensors
            recon_loss = self.criterion.l1_loss(reconstructed, target_images)
            perceptual_loss = self.criterion.perceptual_loss(
                (reconstructed + 1.0) / 2.0, (target_images + 1.0) / 2.0
            )
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
            
            # Recompute total loss with annealed KL weight
            total_loss = (self.criterion.reconstruction_weight * recon_loss + 
                         self.criterion.perceptual_weight * perceptual_loss + 
                         kl_weight * kl_loss)
            
            # Update loss dict with annealed values
            loss_dict = {
                'total_loss': total_loss.item(),
                'reconstruction_loss': recon_loss.item(),
                'perceptual_loss': perceptual_loss.item(),
                'kl_loss': kl_loss.item(),
                'kl_weight': kl_weight
            }
        else:
            loss_dict['kl_weight'] = kl_weight
        
        return total_loss, loss_dict
    
    def denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images from [-1, 1] to [0, 1]."""
        return (images + 1.0) / 2.0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.current_epoch = epoch
        self.vae.train()
        self.text_encoder.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        for batch_idx, batch in enumerate(self.data_loaders['train']):
            images = batch['image'].to(self.device)
            descriptions = batch['full_description']
            
            # Encode text
            with torch.no_grad():
                text_emb = self.text_encoder(descriptions)
            
            # Forward pass through VAE
            outputs = self.vae(images, text_emb, mode='train')
            
            # Compute loss
            loss, loss_dict = self.compute_vae_loss(outputs, images, epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['reconstruction_loss']
            total_kl_loss += loss_dict['kl_loss']
            
            # Log progress
            if batch_idx % self.config['training']['log_every'] == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                               f'Loss: {loss_dict["total_loss"]:.4f}, '
                               f'Recon: {loss_dict["reconstruction_loss"]:.4f}, '
                               f'KL: {loss_dict["kl_loss"]:.4f}, '
                               f'KL Weight: {loss_dict["kl_weight"]:.4f}')
            
            self.global_step += 1
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.vae.eval()
        self.text_encoder.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = len(self.data_loaders['val'])
        
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                images = batch['image'].to(self.device)
                descriptions = batch['full_description']
                
                # Encode text
                text_emb = self.text_encoder(descriptions)
                
                # Forward pass through VAE
                outputs = self.vae(images, text_emb, mode='train')
                
                # Compute loss
                loss, loss_dict = self.compute_vae_loss(outputs, images, epoch)
                
                # Update metrics
                total_loss += loss_dict['total_loss']
                total_recon_loss += loss_dict['reconstruction_loss']
                total_kl_loss += loss_dict['kl_loss']
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate sample images for monitoring."""
        self.vae.eval()
        self.text_encoder.eval()
        
        # Get a batch of validation data
        val_batch = next(iter(self.data_loaders['val']))
        descriptions = val_batch['full_description'][:num_samples]
        real_images = val_batch['image'][:num_samples]
        
        with torch.no_grad():
            # Encode text
            text_emb = self.text_encoder(descriptions)
            
            # Generate reconstructions
            recon_outputs = self.vae(real_images.to(self.device), text_emb, mode='train')
            reconstructed = recon_outputs['reconstructed']
            
            # Generate from prior
            gen_outputs = self.vae(None, text_emb, mode='inference')
            generated = gen_outputs['generated']
        
        # Convert to numpy and denormalize
        real_images = self.denormalize_images(real_images)
        reconstructed = self.denormalize_images(reconstructed)
        generated = self.denormalize_images(generated)
        
        # Create comparison grid
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
        
        for i in range(num_samples):
            # Real image
            axes[0, i].imshow(real_images[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title(f'Real')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title(f'Reconstructed')
            axes[1, i].axis('off')
            
            # Generated image
            axes[2, i].imshow(generated[i].permute(1, 2, 0).cpu().numpy())
            axes[2, i].set_title(f'Generated')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.sample_dir / f'vae_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to tensorboard
        self.tb_writer.add_images('Real_Images', real_images, epoch)
        self.tb_writer.add_images('VAE Reconstructed_Images', reconstructed, epoch)
        self.tb_writer.add_images('VAE Generated_Images', generated, epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f'vae_checkpoint_epoch_{epoch:04d}.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'vae_best_model.pth')
            self.logger.info(f'Saved best model at epoch {epoch}')
        
        # Keep only recent checkpoints
        checkpoints = list(self.checkpoint_dir.glob('vae_checkpoint_epoch_*.pth'))
        if len(checkpoints) > 2:
            checkpoints.sort()
            for old_checkpoint in checkpoints[:-2]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f'Loaded checkpoint from epoch {self.current_epoch}')
    
    def train(self):
        """Main training loop."""
        self.logger.info(f'Starting VAE training for {self.config["training"]["vae_epochs"]} epochs')
        
        for epoch in range(self.current_epoch, self.config['training']['vae_epochs']):
            self.logger.info(f'Epoch {epoch + 1}/{self.config["training"]["vae_epochs"]}')
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            kl_weight = self.get_kl_weight(epoch)
            self.logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                           f'Recon: {train_metrics["recon_loss"]:.4f}, '
                           f'KL: {train_metrics["kl_loss"]:.4f}, '
                           f'KL Weight: {kl_weight:.4f}')
            self.logger.info(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                           f'Recon: {val_metrics["recon_loss"]:.4f}, '
                           f'KL: {val_metrics["kl_loss"]:.4f}')
            
            # TensorBoard logging
            self.tb_writer.add_scalar('VAE Train/Loss', train_metrics['loss'], epoch)
            self.tb_writer.add_scalar('VAE Train/Recon_Loss', train_metrics['recon_loss'], epoch)
            self.tb_writer.add_scalar('VAE Train/KL_Loss', train_metrics['kl_loss'], epoch)
            self.tb_writer.add_scalar('VAE Train/KL_Weight', kl_weight, epoch)
            self.tb_writer.add_scalar('VAE Val/Loss', val_metrics['loss'], epoch)
            self.tb_writer.add_scalar('VAE Val/Recon_Loss', val_metrics['recon_loss'], epoch)
            self.tb_writer.add_scalar('VAE Val/KL_Loss', val_metrics['kl_loss'], epoch)
            
            # Generate samples
            if epoch % self.config['training']['sample_every'] == 0:
                self.generate_samples(epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            self.current_epoch = epoch + 1
        
        self.logger.info('VAE training completed!')
        self.tb_writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Load configuration
    config_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    config = load_config(config_path)
    
    # Create VAE trainer
    trainer = VAETrainer(config, experiment_name="pokemon_vae_stage1")
    
    # Start training
    trainer.train()
