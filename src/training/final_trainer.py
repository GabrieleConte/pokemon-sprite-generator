import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml

try:
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.models.vae_decoder import PokemonVAE, VAEEncoder, VAEDecoder
from src.models.text_encoder import TextEncoder
from src.models.unet import UNet
from src.data import create_data_loaders
from src.utils import get_device


class NoiseScheduler:
    """Noise scheduler for diffusion sampling."""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # Linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Coefficients for reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance calculation
        posterior_variance = self.betas * (1.0 - torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)
        # Clamp to avoid 0 variance
        self.posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean latents according to diffusion schedule."""
        device = x_0.device
        self.to(device)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
    
    def sample_previous_timestep(self, x_t: torch.Tensor, predicted_noise: torch.Tensor, timestep: int) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t) using predicted noise."""
        device = x_t.device
        self.to(device)
        
        # Coefficients for mean
        sqrt_recip_alpha = self.sqrt_recip_alphas[timestep]
        beta = self.betas[timestep]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        # Compute mean
        mean = sqrt_recip_alpha * (x_t - beta * predicted_noise / sqrt_one_minus_alpha_cumprod)
        
        # Add noise if not final step
        if timestep > 0:
            variance = self.posterior_variance[timestep]
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    def to(self, device):
        """Move scheduler to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class FinalPokemonGenerator(nn.Module):
    """
    Final Pokemon generator that combines fine-tuned BERT with pre-trained VAE and U-Net diffusion.
    """
    
    def __init__(self, vae_path: str, diffusion_path: str, text_encoder_config: Dict[str, Any]):
        super().__init__()
        
        # Load pre-trained VAE components
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        
        # VAE Encoder (frozen)
        self.vae_encoder = VAEEncoder(
            input_channels=3,
            latent_dim=text_encoder_config.get('latent_dim', 512)
        )
        
        # VAE Decoder (frozen initially, will be unfrozen for fine-tuning)
        self.vae_decoder = VAEDecoder(
            latent_dim=text_encoder_config.get('latent_dim', 512),
            text_dim=text_encoder_config['text_embedding_dim'],
            output_channels=3
        )
        
        # Load VAE weights
        vae_state_dict = vae_checkpoint['vae_state_dict']
        encoder_state_dict = {}
        decoder_state_dict = {}
        
        for key, value in vae_state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key[8:]] = value
            elif key.startswith('decoder.'):
                decoder_state_dict[key[8:]] = value
        
        self.vae_encoder.load_state_dict(encoder_state_dict)
        self.vae_decoder.load_state_dict(decoder_state_dict)
        
        # Freeze VAE components initially
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        
        # Load pre-trained U-Net
        diffusion_checkpoint = torch.load(diffusion_path, map_location='cpu')
        
        self.unet = UNet(
            latent_dim=text_encoder_config.get('latent_dim', 512),
            text_dim=text_encoder_config['text_embedding_dim'],
            time_emb_dim=text_encoder_config.get('time_emb_dim', 128),
            num_heads=text_encoder_config.get('num_heads', 8)
        )
        
        self.unet.load_state_dict(diffusion_checkpoint['unet_state_dict'])
        
        # Freeze U-Net initially
        for param in self.unet.parameters():
            param.requires_grad = False
        
        # Text encoder (will be fine-tuned)
        self.text_encoder = TextEncoder(
            model_name=text_encoder_config['bert_model'],
            hidden_dim=text_encoder_config['text_embedding_dim'],
            nhead=text_encoder_config['nhead'],
            num_encoder_layers=text_encoder_config['num_encoder_layers']
        )
        
        # Load text encoder weights if available
        if 'text_encoder_state_dict' in vae_checkpoint:
            self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'])
        
        # Diffusion components
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=text_encoder_config.get('num_timesteps', 1000),
            beta_start=text_encoder_config.get('beta_start', 0.0001),
            beta_end=text_encoder_config.get('beta_end', 0.02)
        )
        
        # Latent dimension
        self.latent_dim = text_encoder_config.get('latent_dim', 512)
        
    def forward(self, text_list: List[str], num_inference_steps: int = 50) -> torch.Tensor:
        """
        Generate Pokemon images from text descriptions using diffusion.
        
        Args:
            text_list: List of text descriptions
            num_inference_steps: Number of diffusion steps for sampling
            
        Returns:
            Generated images [batch_size, 3, 215, 215]
        """
        # Encode text
        text_emb = self.text_encoder(text_list)
        
        # Start from pure noise in latent space
        batch_size = text_emb.size(0)
        latent = torch.randn(batch_size, self.latent_dim, 3, 3, device=text_emb.device)
        
        # Diffusion sampling
        step_size = self.noise_scheduler.num_timesteps // num_inference_steps
        
        for i in range(num_inference_steps):
            timestep = self.noise_scheduler.num_timesteps - 1 - i * step_size
            timestep = max(0, timestep)
            
            # Create timestep tensor
            timesteps = torch.full((batch_size,), timestep, device=text_emb.device, dtype=torch.long)
            
            # Predict noise with U-Net
            with torch.no_grad():
                predicted_noise = self.unet(latent, timesteps, text_emb)
            
            # Sample previous timestep
            if timestep > 0:
                latent = self.noise_scheduler.sample_previous_timestep(latent, predicted_noise, timestep)
            else:
                # Final step - just remove predicted noise
                latent = latent - predicted_noise
        
        # Decode latent to image using VAE decoder
        generated = self.vae_decoder(latent, text_emb)
        
        return generated
    
    def unfreeze_vae_decoder(self):
        """Unfreeze VAE decoder for fine-tuning."""
        for param in self.vae_decoder.parameters():
            param.requires_grad = True
    
    def freeze_vae_decoder(self):
        """Freeze VAE decoder."""
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_unet(self):
        """Unfreeze U-Net for fine-tuning."""
        for param in self.unet.parameters():
            param.requires_grad = True
    
    def freeze_unet(self):
        """Freeze U-Net."""
        for param in self.unet.parameters():
            param.requires_grad = False


class FinalTrainer:
    """
    Final trainer that fine-tunes BERT encoder with pre-trained VAE decoder.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 vae_checkpoint_path: str,
                 diffusion_checkpoint_path: str,
                 experiment_name: str = "pokemon_final"):
        """
        Initialize the final trainer.
        
        Args:
            config: Training configuration dictionary
            vae_checkpoint_path: Path to pre-trained VAE checkpoint
            diffusion_checkpoint_path: Path to pre-trained diffusion checkpoint
            experiment_name: Name for the experiment
        """
        self.config = config
        self.vae_checkpoint_path = vae_checkpoint_path
        self.diffusion_checkpoint_path = diffusion_checkpoint_path
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
        self.training_phase = 'text_encoder'  # 'text_encoder' or 'joint'
        
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
                logging.FileHandler(self.log_dir / 'final_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self):
        """Initialize the final model."""
        model_config = self.config['model']
        
        # Create final generator
        self.generator = FinalPokemonGenerator(
            vae_path=self.vae_checkpoint_path,
            diffusion_path=self.diffusion_checkpoint_path,
            text_encoder_config=model_config
        ).to(self.device)
        
        self.logger.info(f"Final generator initialized")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in self.generator.parameters())}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}")
        
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
        
        # Different learning rates for different components
        param_groups = [
            {
                'params': self.generator.text_encoder.parameters(),
                'lr': opt_config['text_encoder_lr'],
                'name': 'text_encoder'
            }
        ]
        
        # Optimizer
        if opt_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                betas=(opt_config['beta1'], opt_config['beta2'])
            )
        elif opt_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=opt_config['weight_decay']
            )
        
        # Learning rate scheduler
        if opt_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['final_epochs']
            )
        elif opt_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=opt_config['step_size'],
                gamma=opt_config['gamma']
            )
        
        # Loss function
        self.criterion = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def setup_monitoring(self):
        """Setup monitoring and logging tools."""
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
    
    def compute_generation_loss(self, generated_images: torch.Tensor, target_images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute generation loss."""
        # Combined L1 and MSE loss
        l1_loss = self.l1_loss(generated_images, target_images)
        mse_loss = self.criterion(generated_images, target_images)
        
        # Weighted combination
        total_loss = l1_loss + 0.1 * mse_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'mse_loss': mse_loss.item()
        }
        
        return total_loss, loss_dict
    
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
            images = batch['image'].to(self.device)
            descriptions = batch['full_description']
            
            # Forward pass
            generated_images = self.generator(descriptions)
            
            # Compute loss
            loss, loss_dict = self.compute_generation_loss(generated_images, images)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss']
            total_l1_loss += loss_dict['l1_loss']
            total_mse_loss += loss_dict['mse_loss']
            
            # Log progress
            if batch_idx % self.config['training']['log_every'] == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                               f'Phase: {self.training_phase}, '
                               f'Loss: {loss_dict["total_loss"]:.4f}, '
                               f'L1: {loss_dict["l1_loss"]:.4f}, '
                               f'MSE: {loss_dict["mse_loss"]:.4f}')
            
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
            for batch in self.data_loaders['val']:
                images = batch['image'].to(self.device)
                descriptions = batch['full_description']
                
                # Forward pass
                generated_images = self.generator(descriptions)
                
                # Compute loss
                loss, loss_dict = self.compute_generation_loss(generated_images, images)
                
                # Update metrics
                total_loss += loss_dict['total_loss']
                total_l1_loss += loss_dict['l1_loss']
                total_mse_loss += loss_dict['mse_loss']
        
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
        
        # Convert to numpy and denormalize
        real_images = self.denormalize_images(real_images)
        generated_images = self.denormalize_images(generated_images)
        
        # Create comparison grid
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        for i in range(num_samples):
            # Real image
            axes[0, i].imshow(real_images[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title('Real')
            axes[0, i].axis('off')
            
            # Generated image
            axes[1, i].imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title('Generated')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.sample_dir / f'final_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to tensorboard
        self.tb_writer.add_images('Real_Images', real_images, epoch)
        self.tb_writer.add_images('Generated_Images', generated_images, epoch)
    
    def switch_to_joint_training(self):
        """Switch to joint training mode (fine-tune both text encoder and VAE decoder)."""
        self.logger.info("Switching to joint training mode - unfreezing VAE decoder")
        self.generator.unfreeze_vae_decoder()
        self.training_phase = 'joint'
        
        # Add VAE decoder parameters to optimizer
        vae_params = {
            'params': self.generator.vae_decoder.parameters(),
            'lr': self.config['optimization']['vae_decoder_lr'],
            'name': 'vae_decoder'
        }
        
        self.optimizer.add_param_group(vae_params)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'training_phase': self.training_phase
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f'final_checkpoint_epoch_{epoch:04d}.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'final_best_model.pth')
            self.logger.info(f'Saved best model at epoch {epoch}')
        
        # Keep only recent checkpoints
        checkpoints = list(self.checkpoint_dir.glob('final_checkpoint_epoch_*.pth'))
        if len(checkpoints) > 5:
            checkpoints.sort()
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_phase = checkpoint.get('training_phase', 'text_encoder')
        
        self.logger.info(f'Loaded checkpoint from epoch {self.current_epoch}')
        self.logger.info(f'Training phase: {self.training_phase}')
    
    def train(self):
        """Main training loop."""
        self.logger.info(f'Starting final training for {self.config["training"]["final_epochs"]} epochs')
        
        # Phase 1: Train only text encoder
        phase1_epochs = self.config['training'].get('phase1_epochs', self.config['training']['final_epochs'] // 2)
        
        for epoch in range(self.current_epoch, self.config['training']['final_epochs']):
            self.logger.info(f'Epoch {epoch + 1}/{self.config["training"]["final_epochs"]}')
            
            # Switch to joint training at specified epoch
            if epoch == phase1_epochs and self.training_phase == 'text_encoder':
                self.switch_to_joint_training()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.logger.info(f'Phase: {self.training_phase}')
            self.logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                           f'L1: {train_metrics["l1_loss"]:.4f}, '
                           f'MSE: {train_metrics["mse_loss"]:.4f}')
            self.logger.info(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                           f'L1: {val_metrics["l1_loss"]:.4f}, '
                           f'MSE: {val_metrics["mse_loss"]:.4f}')
            
            # TensorBoard logging
            self.tb_writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.tb_writer.add_scalar('Train/L1_Loss', train_metrics['l1_loss'], epoch)
            self.tb_writer.add_scalar('Train/MSE_Loss', train_metrics['mse_loss'], epoch)
            self.tb_writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.tb_writer.add_scalar('Val/L1_Loss', val_metrics['l1_loss'], epoch)
            self.tb_writer.add_scalar('Val/MSE_Loss', val_metrics['mse_loss'], epoch)
            
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
        
        self.logger.info('Final training completed!')
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
    
    # Path to pre-trained VAE model
    vae_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/experiments/pokemon_vae_stage1/checkpoints/vae_best_model.pth"
    
    # Path to pre-trained diffusion model
    diffusion_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/experiments/pokemon_diffusion_stage2/checkpoints/diffusion_best_model.pth"
    
    # Create final trainer
    trainer = FinalTrainer(
        config=config,
        vae_checkpoint_path=vae_checkpoint_path,
        diffusion_checkpoint_path=diffusion_checkpoint_path,
        experiment_name="pokemon_final_stage3"
    )
    
    # Start training
    trainer.train()
