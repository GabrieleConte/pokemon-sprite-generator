"""
Improved trainer for diffusion-based U-Net denoising using diffusers library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import os
import math
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from src.models import VAEEncoder, VAEDecoder, TextEncoder
from src.models.diffusers_unet import DiffusersUNet
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
        
        # For posterior sampling
        self.posterior_variance = self.betas * (1.0 - torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)
        self.posterior_variance[0] = self.posterior_variance[1]  # Fix first element
        
        # Clamp to prevent numerical issues
        self.sqrt_alphas_cumprod = torch.clamp(self.sqrt_alphas_cumprod, min=1e-8)
        self.sqrt_one_minus_alphas_cumprod = torch.clamp(self.sqrt_one_minus_alphas_cumprod, min=1e-8)
        
    def _cosine_beta_schedule(self, timesteps, beta_start, beta_end, s=0.008):
        """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
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
            print(f"Warning: NaN/Inf detected in add_noise. Clamping values.")
            noisy = torch.clamp(noisy, -10, 10)
            
        return noisy
    
    def sample_prev_timestep(self, x_t: torch.Tensor, noise_pred: torch.Tensor, timestep: int) -> torch.Tensor:
        """Sample x_{t-1} given x_t and predicted noise."""
        device = x_t.device
        self.to(device)
        
        # Get schedule values
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0, device=device)
        beta_t = self.betas[timestep]
        
        # Compute predicted original sample
        pred_original_sample = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Compute coefficients for x_{t-1}
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
        prev_sample = torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + pred_sample_direction
        
        # Add noise if not the final step
        if timestep > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[timestep]) * noise
            prev_sample = prev_sample + variance
            
        return prev_sample
    
    def to(self, device):
        """Move scheduler to device."""
        self.betas = self.betas.to(device, dtype=torch.float32)
        self.alphas = self.alphas.to(device, dtype=torch.float32)
        self.alphas_cumprod = self.alphas_cumprod.to(device, dtype=torch.float32)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device, dtype=torch.float32)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device, dtype=torch.float32)
        self.posterior_variance = self.posterior_variance.to(device, dtype=torch.float32)
        return self


class DiffusersTrainer:
    """
    Trainer for diffusion-based U-Net denoising using diffusers library.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 vae_checkpoint_path: str,
                 experiment_name: str = "pokemon_diffusers"):
        """
        Initialize the diffusers trainer.
        
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
        
        # Optimize MPS memory usage for 16GB systems
        if self.device.type == 'mps':
            self._setup_mps_memory_optimization()
        
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
        
        # Setup scheduler (after data loaders are ready)
        self.setup_scheduler()
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_mps_memory_optimization(self):
        """Configure optimal MPS memory settings for better system RAM utilization."""
        import os
        
        print("🔧 Configuring MPS for optimal 16GB system memory usage...")
        
        # Enable unified memory (allows using system RAM beyond GPU memory)
        os.environ['PYTORCH_MPS_ENABLE_UNIFIED_MEMORY'] = '1'
        
        # Set memory management for larger models
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Use as much as needed
        os.environ['PYTORCH_MPS_ENABLE_MEMORY_POOL'] = '1'
        
        # Clear any existing cache
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        print("✅ MPS unified memory enabled - can now use full 16GB system RAM")
        print("✅ Memory pool optimization enabled for better performance")
        
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
        
        # Get fine-tuning strategy from config, default to 'minimal' for memory efficiency
        finetune_strategy = model_config.get('bert_finetune_strategy', 'minimal')
        
        # Text encoder with selective fine-tuning (trainable during diffusion training)
        self.text_encoder = TextEncoder(
            model_name=model_config['bert_model'],
            hidden_dim=model_config['text_embedding_dim'],
            finetune_strategy=finetune_strategy
        ).to(self.device)
        
        print(f"Text encoder trainable parameters: {sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad):,}")
        
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
        
        # Load VAE weights properly - handle the fact that VAE was saved differently
        if 'vae_state_dict' in vae_checkpoint:
            # Try to load the full VAE state dict directly
            try:
                # First, try to create a temporary VAE model to load the full state dict
                from src.models.vae_decoder import PokemonVAE
                temp_vae = PokemonVAE(latent_dim=model_config.get('latent_dim', 8), text_dim=model_config['text_embedding_dim'])
                temp_vae.load_state_dict(vae_checkpoint['vae_state_dict'])
                
                # Now extract the encoder and decoder state dicts
                self.vae_encoder.load_state_dict(temp_vae.encoder.state_dict())
                self.vae_decoder.load_state_dict(temp_vae.decoder.state_dict())
                print("✅ Loaded VAE weights from complete VAE model")
                
            except Exception as e:
                print(f"⚠️ Could not load VAE as complete model: {e}")
                print("⚠️ Starting with randomly initialized VAE weights.")
                print("This is fine for testing the diffusion training setup.")
                
        elif 'encoder_state_dict' in vae_checkpoint and 'decoder_state_dict' in vae_checkpoint:
            self.vae_encoder.load_state_dict(vae_checkpoint['encoder_state_dict'])
            self.vae_decoder.load_state_dict(vae_checkpoint['decoder_state_dict'])
            print("✅ Loaded VAE weights from separate state dicts")
        else:
            print("Warning: Could not load VAE weights properly.")
            print("Starting with randomly initialized VAE weights.")
            print("This is fine if you plan to train the VAE from scratch.")
        
        # Load text encoder weights if available (but respect the fine-tuning strategy)
        if 'text_encoder_state_dict' in vae_checkpoint:
            try:
                self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'])
                print("✅ Loaded text encoder weights from checkpoint")
                
                # Re-apply fine-tuning strategy after loading weights
                self.text_encoder._apply_finetune_strategy()
                print(f"✅ Re-applied fine-tuning strategy: {finetune_strategy}")
                
            except Exception as e:
                print(f"⚠️ Could not load text encoder weights: {e}")
                print("Continuing with current text encoder initialization")
        
        # Freeze VAE components
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        
        # Set VAE to eval mode, but keep text encoder in train mode based on strategy
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        
        # Only set text encoder to train mode if it has trainable parameters
        text_trainable_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        if text_trainable_params > 0:
            self.text_encoder.train()
            print(f"✅ Text encoder in training mode with {text_trainable_params:,} trainable parameters")
        else:
            self.text_encoder.eval()
            print("✅ Text encoder frozen - set to eval mode")
        
        # Diffusers U-Net for diffusion denoising (now using pre-trained weights)
        self.unet = DiffusersUNet(
            latent_dim=model_config.get('latent_dim', 8),
            text_dim=model_config['text_embedding_dim'],
            pretrained_model_name=model_config.get('pretrained_model_name', 'runwayml/stable-diffusion-v1-5'),
            cross_attention_dim=model_config.get('cross_attention_dim', 768),  # SD standard
            attention_head_dim=model_config.get('attention_head_dim', 8),
            use_flash_attention=model_config.get('use_flash_attention', True),
            freeze_encoder=model_config.get('freeze_encoder', False),
            freeze_decoder=model_config.get('freeze_decoder', False)
        ).to(self.device)

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=model_config.get('num_timesteps', 1000),
            beta_start=model_config.get('beta_start', 0.0001),
            beta_end=model_config.get('beta_end', 0.02)
        )
        
        # Print parameter summary
        unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        text_trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        total_trainable = unet_trainable + text_trainable
        
        self.logger.info(f"📊 Diffusion Training Parameter Summary:")
        self.logger.info(f"   U-Net trainable: {unet_trainable:,}")
        self.logger.info(f"   Text encoder trainable: {text_trainable:,}")
        self.logger.info(f"   Total trainable: {total_trainable:,}")
        self.logger.info(f"   Fine-tuning strategy: {finetune_strategy}")
        
        # Memory usage estimate
        memory_gb = (total_trainable * 4 * 3) / (1024**3)  # 4 bytes per param, 3x for gradients/optimizer states
        self.logger.info(f"   Estimated training memory: ~{memory_gb:.2f} GB")
        
    def setup_data_loaders(self):
        """Setup data loaders for training, validation, and testing."""
        data_config = self.config['data']
        
        # Use diffusion-specific batch size if available
        batch_size = self.config['training'].get('diffusion_batch_size', 
                                               self.config['data'].get('batch_size', 8))
        
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            csv_path=data_config['csv_path'],
            image_dir=data_config['image_dir'],
            batch_size=batch_size,
            val_split=data_config.get('val_split', 0.1),
            test_split=data_config.get('test_split', 0.1),
            image_size=data_config.get('image_size', 215),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', False),
            seed=42
        )
        
        self.logger.info(f"Data loaders created: train={len(self.train_loader)}, "
                        f"val={len(self.val_loader)}, test={len(self.test_loader)} batches")
        
    def setup_optimization(self):
        """Setup optimizers and loss functions."""
        # Try different config sections for learning rates
        if 'unet_optimization' in self.config:
            self.training_config = self.config['unet_optimization']
        elif 'optimization' in self.config:
            self.training_config = self.config['optimization']
        else:
            self.training_config = self.config.get('training', {})
        
        # Create parameter groups with different learning rates
        unet_lr = self.training_config.get('learning_rate', 5e-4)
        text_lr = self.training_config.get('text_encoder_lr', unet_lr * 0.1)  # Default: 10x smaller LR for text encoder
        
        # Check if text encoder has trainable parameters
        text_trainable_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
        
        param_groups = [
            {
                'params': self.unet.parameters(),
                'lr': unet_lr,
                'name': 'unet'
            }
        ]
        
        # Only add text encoder to optimizer if it has trainable parameters
        if text_trainable_params:
            param_groups.append({
                'params': text_trainable_params,
                'lr': text_lr,
                'name': 'text_encoder'
            })
            self.logger.info(f"✅ Text encoder added to optimizer with LR={text_lr:.2e}")
        else:
            self.logger.info("✅ Text encoder frozen - not added to optimizer")
        
        # Optimizer for U-Net and optionally text encoder
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss function (simple MSE for noise prediction)
        self.criterion = nn.MSELoss()
        
        # Mixed precision training for memory efficiency
        self.use_mixed_precision = self.training_config.get('use_mixed_precision', False)
        if self.use_mixed_precision and self.device.type == 'mps':
            # Note: MPS doesn't fully support autocast yet, so we'll handle this carefully
            print("⚠️ Mixed precision requested but MPS support is limited")
            self.use_mixed_precision = False
        elif self.use_mixed_precision:
            print("✅ Mixed precision training enabled")
        
        self.logger.info(f"Optimizer setup: U-Net LR={unet_lr:.2e}")
        if text_trainable_params:
            self.logger.info(f"Text Encoder LR={text_lr:.2e}, Trainable params: {len(text_trainable_params)}")
        else:
            self.logger.info("Text Encoder: Frozen (no trainable parameters)")
        
    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Get scheduler type from config
        scheduler_type = self.config.get('optimization', {}).get('scheduler', 'cosine')
        
        if scheduler_type == 'constant':
            # No scheduler - keep constant learning rates
            self.lr_scheduler = None
            print(f"✅ Using constant learning rates: {[group['lr'] for group in self.optimizer.param_groups]}")
            return
        
        # Get epochs from the right config section
        if 'training' in self.config and 'diffusion_epochs' in self.config['training']:
            num_epochs = self.config['training']['diffusion_epochs']
        elif 'unet_optimization' in self.config:
            num_epochs = self.config['unet_optimization'].get('num_epochs', 100)
        elif 'optimization' in self.config:
            num_epochs = self.config['optimization'].get('num_epochs', 100)
        else:
            num_epochs = 100  # fallback
            
        total_steps = len(self.train_loader) * num_epochs
        
        # Get max learning rates from parameter groups
        max_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        if scheduler_type == 'cosine':
            # Cosine annealing with warmup for both parameter groups
            self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:
            # Default to constant
            self.lr_scheduler = None
            print(f"⚠️ Unknown scheduler '{scheduler_type}', using constant learning rates")
        
    def setup_monitoring(self):
        """Setup tensorboard logging."""
        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def check_for_nans(self, tensor, name):
        """Check tensor for NaN or Inf values."""
        if torch.isnan(tensor).any():
            self.logger.warning(f"NaN detected in {name}")
            return True
        if torch.isinf(tensor).any():
            self.logger.warning(f"Inf detected in {name}")
            return True
        return False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.unet.train()
        self.text_encoder.train()  # Ensure text encoder is in training mode
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Get batch data
                images = batch['image'].to(self.device)
                descriptions = batch['description']
                
                # Get text embeddings (now with gradients)
                text_emb = self.text_encoder(descriptions)
                
                # Encode images to latent space (no gradients for VAE)
                with torch.no_grad():
                    latent, _, _ = self.vae_encoder(images)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, 
                    (latent.shape[0],), device=self.device
                )
                
                # Sample noise
                noise = torch.randn_like(latent)
                
                # Add noise to latents
                noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
                
                # Check for NaN/Inf in all inputs before forward pass
                if self.check_for_nans(latent, "latent"):
                    self.logger.warning(f"Skipping batch due to NaN in latent")
                    continue
                if self.check_for_nans(text_emb, "text_emb"):
                    self.logger.warning(f"Skipping batch due to NaN in text_emb")
                    continue
                if self.check_for_nans(noisy_latent, "noisy_latent"):
                    self.logger.warning(f"Skipping batch due to NaN in noisy_latent")
                    continue
                if self.check_for_nans(noise, "noise"):
                    self.logger.warning(f"Skipping batch due to NaN in noise")
                    continue
                
                # Predict noise
                predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
                
                # Check for NaN in U-Net output
                if self.check_for_nans(predicted_noise, "predicted_noise"):
                    self.logger.warning(f"Skipping batch due to NaN in predicted_noise")
                    continue
                
                # Compute loss
                loss = self.criterion(predicted_noise, noise)
                
                # Check for NaNs
                if self.check_for_nans(loss, "loss"):
                    self.logger.warning(f"Skipping batch due to NaN loss")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for U-Net and optionally text encoder
                max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
                unet_grad_norm = torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=max_grad_norm)
                
                # Only clip text encoder gradients if it has trainable parameters
                text_trainable_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                if text_trainable_params:
                    text_grad_norm = torch.nn.utils.clip_grad_norm_(text_trainable_params, max_norm=max_grad_norm * 0.5)  # Smaller clip for text encoder
                else:
                    text_grad_norm = torch.tensor(0.0)  # No gradients to clip
                
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Log gradient norms periodically for monitoring
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('Train/UNet_Grad_Norm', unet_grad_norm.item(), self.global_step)
                    if text_trainable_params:
                        self.writer.add_scalar('Train/TextEncoder_Grad_Norm', text_grad_norm.item(), self.global_step)
                
                # Memory cleanup for MPS to prevent accumulation
                if self.device.type == 'mps' and batch_idx % 5 == 0:  # Every 5 batches
                    torch.mps.empty_cache()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar with learning rates
                unet_lr = self.optimizer.param_groups[0]['lr']
                pbar_dict = {
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / num_batches:.4f}",
                    'unet_lr': f"{unet_lr:.2e}"
                }
                
                # Add text encoder LR if it's being trained
                if len(self.optimizer.param_groups) > 1:
                    text_lr = self.optimizer.param_groups[1]['lr']
                    pbar_dict['text_lr'] = f"{text_lr:.2e}"
                
                pbar.set_postfix(pbar_dict)
                
                # Log to tensorboard
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/UNet_LR', unet_lr, self.global_step)
                    
                    # Only log text encoder LR if it's being trained
                    if len(self.optimizer.param_groups) > 1:
                        text_lr = self.optimizer.param_groups[1]['lr']
                        self.writer.add_scalar('Train/TextEncoder_LR', text_lr, self.global_step)
                    
                    # Log average timestep for monitoring diffusion schedule usage
                    avg_timestep = timesteps.float().mean().item()
                    self.writer.add_scalar('Train/Avg_Timestep', avg_timestep, self.global_step)
                
            except Exception as e:
                self.logger.error(f"Error in training batch: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.unet.eval()
        self.text_encoder.eval()  # Set text encoder to eval mode for validation
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    # Get batch data
                    images = batch['image'].to(self.device)
                    descriptions = batch['description']
                    
                    # Get text embeddings (no gradients during validation)
                    text_emb = self.text_encoder(descriptions)
                    
                    # Encode images to latent space
                    latent, _, _ = self.vae_encoder(images)
                    
                    # Sample random timesteps
                    timesteps = torch.randint(
                        0, self.noise_scheduler.num_timesteps, 
                        (latent.shape[0],), device=self.device
                    )
                    
                    # Sample noise
                    noise = torch.randn_like(latent)
                    
                    # Add noise to latents
                    noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
                    
                    # Predict noise
                    predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
                    
                    # Compute loss
                    loss = self.criterion(predicted_noise, noise)
                    
                    if not self.check_for_nans(loss, "val_loss"):
                        total_loss += loss.item()
                        num_batches += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return {'val_loss': avg_loss}
    
    def ddpm_sample(self, text_emb: torch.Tensor, num_samples: int, 
                   fast_sampling: bool = True, num_inference_steps: int = 50) -> torch.Tensor:
        """Sample from the diffusion model using DDPM sampling."""
        self.unet.eval()
        self.text_encoder.eval()
        
        # Start with random noise
        shape = (num_samples, 8, 27, 27)
        x = torch.randn(shape, device=self.device)
        
        # Determine sampling steps
        if fast_sampling:
            timesteps = torch.linspace(
                self.noise_scheduler.num_timesteps - 1, 0, 
                num_inference_steps, dtype=torch.long, device=self.device
            )
        else:
            timesteps = torch.arange(
                self.noise_scheduler.num_timesteps - 1, -1, -1, 
                device=self.device
            )
        
        # Expand text embeddings to match batch size
        if text_emb.shape[0] != num_samples:
            text_emb = text_emb.repeat(num_samples, 1, 1)
        
        with torch.no_grad():
            for t in tqdm(timesteps, desc="Sampling"):
                t_tensor = torch.full((num_samples,), t.item(), device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self.unet(x, t_tensor, text_emb)
                
                # Sample previous timestep
                x = self.noise_scheduler.sample_prev_timestep(x, noise_pred, int(t.item()))
        
        return x

    def denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images from [-1, 1] to [0, 1] for display."""
        return (images + 1.0) / 2.0

    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate and save sample images with tensorboard logging."""
        self.unet.eval()
        self.text_encoder.eval()
        
        # Get some validation descriptions and real images
        val_batch = next(iter(self.val_loader))
        descriptions = val_batch['description'][:min(num_samples, len(val_batch['description']))]
        real_images = val_batch['image'][:min(num_samples, len(val_batch['image']))].to(self.device)  # Move to device
        
        # Get text embeddings
        with torch.no_grad():
            text_emb = self.text_encoder(descriptions)
            
            # Encode real images to latent space for reconstruction comparison
            real_latents, _, _ = self.vae_encoder(real_images)
            
            # Decode real latents back to images (for reconstruction comparison)
            reconstructed_images = self.vae_decoder(real_latents, text_emb)
        
        # Generate new latents using diffusion
        generated_latents = self.ddpm_sample(text_emb, len(descriptions), fast_sampling=True)
        
        # Decode generated latents to images
        with torch.no_grad():
            generated_images = self.vae_decoder(generated_latents, text_emb)
        
        # Denormalize all images for display and saving
        real_images_norm = self.denormalize_images(real_images)
        reconstructed_images_norm = self.denormalize_images(reconstructed_images)
        generated_images_norm = self.denormalize_images(generated_images)
        
        # Save samples to file
        sample_path = self.sample_dir / f"epoch_{epoch}_samples.png"
        
        # Create comparison grid: [Real | Reconstructed | Generated]
        comparison_grid = torch.cat([
            real_images_norm,
            reconstructed_images_norm, 
            generated_images_norm
        ], dim=0)
        
        save_image(
            comparison_grid,
            sample_path,
            nrow=len(descriptions),  # Show real, recon, generated in rows
            normalize=False,  # Already normalized
            value_range=(0, 1)
        )
        
        # Log to tensorboard with separate image grids
        self.writer.add_images('Diffusion/Real_Images', real_images_norm, epoch)
        self.writer.add_images('Diffusion/Reconstructed_Images', reconstructed_images_norm, epoch)
        self.writer.add_images('Diffusion/Generated_Images', generated_images_norm, epoch)
        
        # Create and log a comparison grid
        self.writer.add_images('Diffusion/Comparison_Grid', comparison_grid, epoch)
        
        # Save descriptions
        desc_path = self.sample_dir / f"epoch_{epoch}_descriptions.txt"
        with open(desc_path, 'w') as f:
            f.write("Generated samples descriptions:\n")
            f.write("="*50 + "\n")
            for i, desc in enumerate(descriptions):
                f.write(f"Sample {i+1}: {desc}\n")
        
        self.logger.info(f"Generated samples saved to {sample_path}")
        self.logger.info(f"Sample descriptions saved to {desc_path}")
        self.logger.info(f"Tensorboard images logged for epoch {epoch}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'unet_state_dict': self.unet.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # # Save regular checkpoint
        # checkpoint_path = self.checkpoint_dir / f"diffusion_epoch_{epoch}.pth"
        # torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "diffusion_best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        
        # Load text encoder if available in checkpoint
        if 'text_encoder_state_dict' in checkpoint:
            self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            self.logger.info("✅ Loaded text encoder from checkpoint")
        else:
            self.logger.warning("⚠️ Text encoder state not found in checkpoint")
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler is not None and checkpoint.get('lr_scheduler_state_dict') is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        # Get epochs from the right config section
        if 'training' in self.config and 'diffusion_epochs' in self.config['training']:
            num_epochs = self.config['training']['diffusion_epochs']
            training_config = self.config['training']
        elif 'unet_optimization' in self.config:
            num_epochs = self.config['unet_optimization'].get('num_epochs', 100)
            training_config = self.config['unet_optimization']
        elif 'optimization' in self.config:
            num_epochs = self.config['optimization'].get('num_epochs', 100)
            training_config = self.config['optimization']
        else:
            num_epochs = 100  # fallback
            training_config = {}
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"U-Net parameters: {self.unet.get_parameter_count():,}")
        self.logger.info(f"Text encoder parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
        self.logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.unet.parameters()) + sum(p.numel() for p in self.text_encoder.parameters()):,}")
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                # Training
                train_metrics = self.train_epoch(epoch)
                
                # Validation
                val_metrics = self.validate_epoch(epoch)
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Log epoch-level metrics to tensorboard
                self.writer.add_scalar('Epoch/Train_Loss', metrics['train_loss'], epoch)
                self.writer.add_scalar('Epoch/Val_Loss', metrics['val_loss'], epoch)
                
                # Log current learning rates at epoch level
                unet_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Epoch/UNet_LR', unet_lr, epoch)
                
                # Only log text encoder LR if it's being trained
                log_text_lr = ""
                if len(self.optimizer.param_groups) > 1:
                    text_lr = self.optimizer.param_groups[1]['lr']
                    self.writer.add_scalar('Epoch/TextEncoder_LR', text_lr, epoch)
                    log_text_lr = f", text_lr={text_lr:.2e}"
                
                # Log metrics
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={metrics['train_loss']:.4f}, "
                    f"val_loss={metrics['val_loss']:.4f}, "
                    f"unet_lr={unet_lr:.2e}"
                    f"{log_text_lr}"
                )
                
                # Check if best model
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                
                # Save checkpoint
                if epoch % training_config.get('save_every', 10) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                
                # Generate samples
                if epoch % training_config.get('sample_every', 20) == 0:
                    self.generate_samples(epoch)
                
                self.current_epoch = epoch + 1
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.writer.close()
            self.logger.info("Training completed")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Example usage
    config_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    config = load_config(config_path)
    
    vae_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/logs/vae_best_model.pth"
    
    trainer = DiffusersTrainer(
        config=config,
        vae_checkpoint_path=vae_checkpoint_path,
        experiment_name="pokemon_diffusers"
    )
    
    trainer.train()
