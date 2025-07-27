"""
Improved trainer for diffusion-based U-Net denoising with numerical stability fixes.
"""

import torch
import torch.nn as nn
import logging
import yaml
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from src.models.vae_decoder import VAEEncoder, VAEDecoder
from src.models.text_encoder import TextEncoder
from src.models.unet import UNet
from src.data import create_data_loaders
from src.utils import get_device


class NoiseScheduler:
    """Improved noise scheduler with better numerical stability."""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # Cosine schedule for better training stability
        self.betas = self._cosine_beta_schedule(num_timesteps, beta_start, beta_end).float()
        self.alphas = (1.0 - self.betas).float()
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).float()
        
        # Precompute values for sampling with numerical stability
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).float()
        
        # Clamp to prevent numerical issues
        self.sqrt_alphas_cumprod = torch.clamp(self.sqrt_alphas_cumprod, min=1e-8)
        self.sqrt_one_minus_alphas_cumprod = torch.clamp(self.sqrt_one_minus_alphas_cumprod, min=1e-8)
        
    def _cosine_beta_schedule(self, timesteps, beta_start, beta_end, s=0.008):
        """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)  # Use float32 instead of float64
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
        
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean latents according to diffusion schedule."""
        device = x_0.device
        self.to(device)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noisy = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        # Check for NaN/Inf and handle
        if torch.isnan(noisy).any() or torch.isinf(noisy).any():
            print("Warning: NaN/Inf detected in noise addition, using fallback")
            return x_0 + 0.1 * noise  # Simple fallback
            
        return noisy
    
    def to(self, device):
        """Move scheduler to device."""
        self.betas = self.betas.to(device, dtype=torch.float32)
        self.alphas = self.alphas.to(device, dtype=torch.float32)
        self.alphas_cumprod = self.alphas_cumprod.to(device, dtype=torch.float32)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device, dtype=torch.float32)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device, dtype=torch.float32)
        return self


class ImprovedDiffusionTrainer:
    """
    Improved trainer for diffusion-based U-Net denoising with stability fixes.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 vae_checkpoint_path: str,
                 experiment_name: str = "pokemon_diffusion"):
        """
        Initialize the improved diffusion trainer.
        
        Args:
            config: Training configuration dictionary
            vae_checkpoint_path: Path to pre-trained VAE checkpoint
            experiment_name: Name for the experiment
        """
        self.config = config
        self.vae_checkpoint_path = vae_checkpoint_path
        self.experiment_name = experiment_name
        
        # Setup device
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Setup models
        self.setup_models()
        
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
                logging.FileHandler(self.log_dir / 'diffusion_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """Initialize models and load pre-trained VAE."""
        model_config = self.config['model']
        
        # Text encoder (frozen during diffusion training)
        self.text_encoder = TextEncoder(
            model_name=model_config['bert_model'],
            hidden_dim=model_config['text_embedding_dim']
        ).to(self.device)
        
        # Load pre-trained VAE
        print(f"Loading VAE from {self.vae_checkpoint_path}")
        vae_checkpoint = torch.load(self.vae_checkpoint_path, map_location=self.device)
        
        # VAE Encoder and Decoder (frozen during diffusion training)
        self.vae_encoder = VAEEncoder(
            input_channels=3,
            latent_dim=model_config.get('latent_dim', 8)
        ).to(self.device)
        
        self.vae_decoder = VAEDecoder(
            latent_dim=model_config.get('latent_dim', 8),
            text_dim=model_config['text_embedding_dim'],
            output_channels=3
        ).to(self.device)
        
        # Load VAE weights properly
        if 'vae_state_dict' in vae_checkpoint:
            vae_state_dict = vae_checkpoint['vae_state_dict']
            
            # Extract encoder and decoder state dicts
            encoder_state_dict = {}
            decoder_state_dict = {}
            
            for key, value in vae_state_dict.items():
                if key.startswith('encoder.'):
                    encoder_state_dict[key[8:]] = value  # Remove 'encoder.' prefix
                elif key.startswith('decoder.'):
                    decoder_state_dict[key[8:]] = value  # Remove 'decoder.' prefix
            
            self.vae_encoder.load_state_dict(encoder_state_dict, strict=False)
            self.vae_decoder.load_state_dict(decoder_state_dict, strict=False)
        
        # Load text encoder weights if available
        if 'text_encoder_state_dict' in vae_checkpoint:
            self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'], strict=False)
        
        # Freeze VAE and text encoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        self.text_encoder.eval()
        
        # U-Net for diffusion denoising
        self.unet = UNet(
            latent_dim=model_config.get('latent_dim', 8),
            text_dim=model_config['text_embedding_dim'],
            time_emb_dim=model_config.get('time_emb_dim', 128),
            num_heads=model_config.get('num_heads', 4)  # Reduced for stability
        ).to(self.device)

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=model_config.get('num_timesteps', 1000),
            beta_start=model_config.get('beta_start', 0.0001),
            beta_end=model_config.get('beta_end', 0.02)
        )
        
        self.logger.info(f"U-Net initialized with {sum(p.numel() for p in self.unet.parameters())} parameters")
        
    def setup_data_loaders(self):
        """Setup data loaders for training, validation, and testing."""
        data_config = self.config['data']
        
        # Use U-Net specific batch size if available
        unet_config = self.config.get('unet_optimization', {})
        batch_size = unet_config.get('batch_size', data_config['batch_size'])
        num_workers = unet_config.get('num_workers', data_config['num_workers'])
        
        train_loader, val_loader, test_loader = create_data_loaders(
            csv_path=data_config['csv_path'],
            image_dir=data_config['image_dir'],
            batch_size=batch_size,  # Use U-Net specific batch size
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            image_size=data_config['image_size'],
            num_workers=num_workers,  # Use U-Net specific num_workers
            pin_memory=data_config['pin_memory']
        )
        
        self.data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        self.logger.info(f"Data loaders created with U-Net config: batch_size={batch_size}, num_workers={num_workers}")
        self.logger.info(f"Data loaders created: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler using U-Net specific config."""
        # Use U-Net specific optimization config if available, fallback to general config
        unet_config = self.config.get('unet_optimization', {})
        opt_config = self.config['optimization']
        
        # Get U-Net specific parameters with fallbacks
        optimizer_type = unet_config.get('optimizer', opt_config['optimizer'])
        lr = unet_config.get('learning_rate', opt_config['learning_rate'])
        beta1 = unet_config.get('beta1', opt_config['beta1'])
        beta2 = unet_config.get('beta2', opt_config['beta2'])
        weight_decay = unet_config.get('weight_decay', opt_config['weight_decay'])
        max_grad_norm = unet_config.get('max_grad_norm', opt_config['max_grad_norm'])
        
        self.logger.info(f"Using U-Net optimization: {optimizer_type}, lr={lr}, wd={weight_decay}, clip={max_grad_norm}")
        
        # Store gradient clipping norm for use in training
        self.max_grad_norm = max_grad_norm
        
        # Optimizer (only for U-Net parameters)
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
                eps=1e-6  # Smaller eps for stability
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.unet.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
                eps=1e-6  # Smaller eps for stability
            )
        
        # Learning rate scheduler with warmup
        scheduler_type = unet_config.get('scheduler', opt_config.get('scheduler', 'cosine'))
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=self.config['training']['diffusion_epochs'] * 100,  # Estimate
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos'
            )
        else:
            # Fallback to constant LR
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        
        # Improved loss function with Huber loss for stability
        self.criterion = nn.SmoothL1Loss(beta=0.1)  # More stable than MSE
        
    def setup_monitoring(self):
        """Setup monitoring with TensorBoard."""
        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def check_for_nans(self, tensor, name):
        """Check for NaN values and log warnings."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.logger.warning(f"NaN/Inf detected in {name}")
            return True
        return False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.unet.train()
        
        total_loss = 0.0
        num_batches = 0
        nan_count = 0
        
        pbar = tqdm(self.data_loaders['train'], desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(self.device)
                descriptions = batch['full_description']
                
                # Encode text
                with torch.no_grad():
                    text_emb = self.text_encoder(descriptions)
                    if self.check_for_nans(text_emb, "text_emb"):
                        continue
                
                # Encode images to latent space
                with torch.no_grad():
                    latent, mu, logvar = self.vae_encoder(images)
                    if self.check_for_nans(latent, "latent"):
                        continue
                
                # Normalize latent to reasonable range
                latent = torch.clamp(latent, -3.0, 3.0)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, 
                    (images.shape[0],), 
                    device=self.device
                )
                
                # Add noise to latents - use standard noise scale
                noise = torch.randn_like(latent)  # Normal noise scale for proper diffusion
                noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
                
                if self.check_for_nans(noisy_latent, "noisy_latent"):
                    continue
                
                # Predict noise with U-Net
                self.optimizer.zero_grad()
                predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
                
                if self.check_for_nans(predicted_noise, "predicted_noise"):
                    nan_count += 1
                    continue
                
                # Compute loss (predict the noise)
                loss = self.criterion(predicted_noise, noise)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected, skipping batch")
                    nan_count += 1
                    continue
                
                # Backward pass with gradient scaling
                loss.backward()
                
                # Check for gradient explosion
                total_norm = 0
                for p in self.unet.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                if total_norm > self.max_grad_norm * 2:  # Gradient explosion threshold
                    self.logger.warning(f"Large gradient norm: {total_norm:.3f}, clipping to {self.max_grad_norm}")
                
                # Gradient clipping using U-Net specific parameter
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'nans': nan_count
                })
                
                # Log to TensorBoard
                if batch_idx % self.config['training']['log_every'] == 0:
                    self.writer.add_scalar('Diffusion Train/Loss', loss.item(), self.global_step)
                    self.writer.add_scalar('Diffusion Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar('Diffusion Train/Gradient_Norm', total_norm, self.global_step)
                    
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            self.logger.error("No valid batches processed!")
            return {'train_loss': float('inf')}
            
        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.6f}, NaN batches = {nan_count}")
        
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.unet.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Validation"):
                try:
                    images = batch['image'].to(self.device)
                    descriptions = batch['full_description']
                    
                    # Encode text
                    text_emb = self.text_encoder(descriptions)
                    
                    # Encode images to latent space
                    latent, mu, logvar = self.vae_encoder(images)
                    latent = torch.clamp(latent, -3.0, 3.0)
                    
                    # Sample random timesteps
                    timesteps = torch.randint(
                        0, self.noise_scheduler.num_timesteps, 
                        (images.shape[0],), 
                        device=self.device
                    )
                    
                    # Add noise to latents - use standard noise scale for validation too
                    noise = torch.randn_like(latent)
                    noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
                    
                    # Predict noise with U-Net
                    predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
                    
                    # Check for NaN
                    if self.check_for_nans(predicted_noise, "val_predicted_noise"):
                        continue
                    
                    # Compute loss
                    loss = self.criterion(predicted_noise, noise)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        if num_batches == 0:
            return {'val_loss': float('inf')}
            
        avg_loss = total_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Diffusion Val/Loss', avg_loss, epoch)
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # # Save regular checkpoint
        # checkpoint_path = self.checkpoint_dir / f"diffusion_epoch_{epoch}.pth"
        # torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "diffusion_best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
            
        # self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting diffusion training...")
        
        for epoch in range(self.current_epoch, self.config['training']['diffusion_epochs']):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch(epoch)
            
            # Skip validation if training failed
            if train_metrics['train_loss'] == float('inf'):
                self.logger.error(f"Training failed at epoch {epoch}, stopping")
                break
            
            # Validation epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.logger.info("diffusion training completed!")
        self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Example usage
    config_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    config = load_config(config_path)
    
    vae_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/logs/vae_best_model_2.pth"
    
    trainer = ImprovedDiffusionTrainer(
        config=config,
        vae_checkpoint_path=vae_checkpoint_path,
        experiment_name="pokemon_diffusion_improved"
    )
    
    trainer.train()
