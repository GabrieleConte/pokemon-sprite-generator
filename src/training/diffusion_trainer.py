"""
Trainer for diffusion-based U-Net denoising in Pokemon sprite generation.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any
import logging
import yaml
from tqdm import tqdm
from torchvision import transforms

from src.models.vae_decoder import VAEEncoder, VAEDecoder
from src.models.text_encoder import TextEncoder
from src.models.unet import UNet
from src.data import create_data_loaders
from src.utils import get_device


class NoiseScheduler:
    """Improved noise scheduler for diffusion training."""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # Linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean latents according to diffusion schedule."""
        device = x_0.device
        self.to(device)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
    
    def to(self, device):
        """Move scheduler to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self


class DiffusionTrainer:
    """
    Trainer for diffusion-based U-Net denoising.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 vae_checkpoint_path: str,
                 experiment_name: str = "pokemon_diffusion"):
        """
        Initialize the diffusion trainer.
        
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
            hidden_dim=model_config['text_embedding_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers']
        ).to(self.device)
        
        # Load pre-trained VAE
        print(f"Loading VAE from {self.vae_checkpoint_path}")
        vae_checkpoint = torch.load(self.vae_checkpoint_path, map_location=self.device)
        
        # VAE Encoder and Decoder (frozen during diffusion training)
        self.vae_encoder = VAEEncoder(
            input_channels=3,
            latent_dim=model_config.get('latent_dim', 512)
        ).to(self.device)
        
        self.vae_decoder = VAEDecoder(
            latent_dim=model_config.get('latent_dim', 512),
            text_dim=model_config['text_embedding_dim'],
            output_channels=3
        ).to(self.device)
        
        # Load VAE weights from the complete VAE model
        vae_state_dict = vae_checkpoint['vae_state_dict']
        
        # Extract encoder and decoder state dicts from the complete VAE
        encoder_state_dict = {}
        decoder_state_dict = {}
        
        for key, value in vae_state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key[8:]] = value  # Remove 'encoder.' prefix
            elif key.startswith('decoder.'):
                decoder_state_dict[key[8:]] = value  # Remove 'decoder.' prefix
        
        self.vae_encoder.load_state_dict(encoder_state_dict)
        self.vae_decoder.load_state_dict(decoder_state_dict)
        
        # Load text encoder weights if available
        if 'text_encoder_state_dict' in vae_checkpoint:
            self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'])
        
        # Freeze VAE and text encoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # U-Net for denoising
        self.unet = UNet(
            latent_dim=model_config.get('latent_dim', 512),
            text_dim=model_config['text_embedding_dim'],
            time_emb_dim=model_config.get('time_emb_dim', 128),
            num_heads=model_config.get('num_heads', 8)
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
        
        self.logger.info(f"Data loaders created: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        opt_config = self.config['optimization']
        
        # Optimizer (only for U-Net parameters)
        if opt_config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=opt_config['learning_rate'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.unet.parameters(),
                lr=opt_config['learning_rate'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        
        # Learning rate scheduler
        if opt_config['scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['diffusion_epochs']
            )
        elif opt_config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=opt_config['step_size'],
                gamma=opt_config['gamma']
            )
        else:
            self.scheduler = None
        
        # Loss function (MSE for denoising)
        self.criterion = nn.MSELoss()
        
    def setup_monitoring(self):
        """Setup monitoring with TensorBoard."""
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.unet.train()
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        self.text_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.data_loaders['train'], desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            descriptions = batch['description']
            
            # Encode text
            with torch.no_grad():
                text_emb = self.text_encoder(descriptions)
            
            # Encode images to latent space
            with torch.no_grad():
                latent, mu, logvar = self.vae_encoder(images)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.num_timesteps, 
                (images.shape[0],), 
                device=self.device
            )
            
            # Add noise to latents
            noise = torch.randn_like(latent)
            noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
            
            # Predict noise with U-Net
            self.optimizer.zero_grad()
            predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
            
            # Compute loss (predict the noise)
            loss = self.criterion(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if batch_idx % self.config['training']['log_every'] == 0:
                self.writer.add_scalar('Diffusion Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Diffusion Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.unet.eval()
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        self.text_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Validation"):
                images = batch['image'].to(self.device)
                descriptions = batch['description']
                
                # Encode text
                text_emb = self.text_encoder(descriptions)
                
                # Encode images to latent space
                latent, mu, logvar = self.vae_encoder(images)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, 
                    (images.shape[0],), 
                    device=self.device
                )
                
                # Add noise to latents
                noise = torch.randn_like(latent)
                noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
                
                # Predict noise with U-Net
                predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
                
                # Compute loss
                loss = self.criterion(predicted_noise, noise)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Diffusion Val/Loss', avg_loss, epoch)
        
        return {'val_loss': avg_loss}
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate sample images for monitoring."""
        self.unet.eval()
        self.vae_decoder.eval()
        self.text_encoder.eval()
        
        # Sample descriptions from validation set
        val_batch = next(iter(self.data_loaders['val']))
        sample_descriptions = val_batch['description'][:num_samples]
        
        with torch.no_grad():
            # Encode text
            text_emb = self.text_encoder(sample_descriptions)
            
            # Start from pure noise
            latent_shape = (num_samples, self.config['model'].get('latent_dim', 512), 3, 3)
            latent = torch.randn(latent_shape, device=self.device)
            
            # Simple denoising (in practice, you'd use DDPM sampling)
            for t in reversed(range(0, self.noise_scheduler.num_timesteps, 50)):
                timesteps = torch.full((num_samples,), t, device=self.device)
                predicted_noise = self.unet(latent, timesteps, text_emb)
                
                # Simple denoising step (not full DDPM)
                alpha = self.noise_scheduler.alphas[t]
                alpha_cumprod = self.noise_scheduler.alphas_cumprod[t]
                
                latent = (latent - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
            
            # Decode to images
            generated_images = self.vae_decoder(latent, text_emb)
            
            # Denormalize and save
            generated_images = (generated_images + 1.0) / 2.0
            generated_images = torch.clamp(generated_images, 0, 1)
            
            # Save samples
            for i, (img, desc) in enumerate(zip(generated_images, sample_descriptions)):
                img_pil = transforms.ToPILImage()(img.cpu())
                img_pil.save(self.sample_dir / f"epoch_{epoch}_sample_{i}.png")
                
                # Log to TensorBoard
                self.writer.add_image(f'Diffusion Generated/Sample_{i}', img.cpu(), epoch)
                
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"diffusion_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "diffusion_best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
            
        checkpoints = list(self.checkpoint_dir.glob('diffusion_epoch_*.pth'))
        if len(checkpoints) > 2:
            checkpoints.sort()
            for old_checkpoint in checkpoints[:-2]:
                old_checkpoint.unlink()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
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
            
            # Validation epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Generate samples
            if epoch % self.config['training']['sample_every'] == 0:
                self.generate_samples(epoch)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Logging
            self.logger.info(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                           f"val_loss={val_metrics['val_loss']:.4f}")
        
        self.logger.info("Diffusion training completed!")
        
        # Close monitoring
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
    
    vae_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/experiments/pokemon_vae_stage1/checkpoints/vae_best_model.pth"
    
    trainer = DiffusionTrainer(
        config=config,
        vae_checkpoint_path=vae_checkpoint_path,
        experiment_name="pokemon_diffusion_stage3"
    )
    
    trainer.train()
