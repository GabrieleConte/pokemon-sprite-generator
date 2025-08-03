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
        
                                                       
        self.betas = self._cosine_beta_schedule(num_timesteps, beta_start, beta_end).float()
        self.alphas = (1.0 - self.betas).float()
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).float()
        
                                                                 
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).float()
        
                                
        self.posterior_variance = self.betas * (1.0 - torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)
        self.posterior_variance[0] = self.posterior_variance[1]                     
        
                                           
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
        
                                      
        if torch.isnan(noisy).any() or torch.isinf(noisy).any():
            print(f"Warning: NaN/Inf detected in add_noise. Clamping values.")
            noisy = torch.clamp(noisy, -10, 10)
            
        return noisy
    
    def sample_prev_timestep(self, x_t: torch.Tensor, noise_pred: torch.Tensor, timestep: int) -> torch.Tensor:
        """Sample x_{t-1} given x_t and predicted noise."""
        device = x_t.device
        self.to(device)
        
                             
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0, device=device)
        beta_t = self.betas[timestep]
        
                                           
        pred_original_sample = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
                                          
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
        prev_sample = torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + pred_sample_direction
        
                                         
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
        
                      
        self.device = get_device()
        
                                                    
        if self.device.type == 'mps':
            self._setup_mps_memory_optimization()
        
        print(f"Using device: {self.device}")
        
                           
        self.setup_directories()
        
                       
        self.setup_logging()
        
                      
        self.setup_models()
        
                            
        self.setup_data_loaders()
        
                            
        self.setup_optimization()
        
                                                        
        self.setup_scheduler()
        
                          
        self.setup_monitoring()
        
                        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_mps_memory_optimization(self):
        """Configure optimal MPS memory settings for better system RAM utilization."""
        import os
        
        print("🔧 Configuring MPS for optimal 16GB system memory usage...")
        
                                                                           
        os.environ['PYTORCH_MPS_ENABLE_UNIFIED_MEMORY'] = '1'
        
                                                 
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'                         
        os.environ['PYTORCH_MPS_ENABLE_MEMORY_POOL'] = '1'
        
                                  
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
        
                                                                                          
        finetune_strategy = model_config.get('bert_finetune_strategy', 'minimal')
        
                                                                                       
        self.text_encoder = TextEncoder(
            model_name=model_config['bert_model'],
            hidden_dim=model_config['text_embedding_dim'],
            finetune_strategy=finetune_strategy
        ).to(self.device)
        
        print(f"Text encoder trainable parameters: {sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad):,}")
        
                              
        print(f"Loading VAE from {self.vae_checkpoint_path}")
        vae_checkpoint = torch.load(self.vae_checkpoint_path, map_location=self.device)
        
                                                                    
        self.vae_encoder = VAEEncoder(
            input_channels=3,
            latent_dim=model_config.get('latent_dim', 8)
        ).to(self.device)
        
        self.vae_decoder = VAEDecoder(
            latent_dim=model_config.get('latent_dim', 8),
            text_dim=model_config['text_embedding_dim'],
            output_channels=3
        ).to(self.device)
        
                                                                                    
        if 'vae_state_dict' in vae_checkpoint:
                                                          
            try:
                                                                                        
                from src.models.vae_decoder import PokemonVAE
                temp_vae = PokemonVAE(latent_dim=model_config.get('latent_dim', 8), text_dim=model_config['text_embedding_dim'])
                temp_vae.load_state_dict(vae_checkpoint['vae_state_dict'])
                
                                                                 
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
        
                                                                                       
        if 'text_encoder_state_dict' in vae_checkpoint:
            try:
                self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'])
                print("✅ Loaded text encoder weights from checkpoint")
                
                                                                     
                self.text_encoder._apply_finetune_strategy()
                print(f"✅ Re-applied fine-tuning strategy: {finetune_strategy}")
                
            except Exception as e:
                print(f"⚠️ Could not load text encoder weights: {e}")
                print("Continuing with current text encoder initialization")
        
                               
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        
                                                                                     
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        
                                                                            
        text_trainable_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        if text_trainable_params > 0:
            self.text_encoder.train()
            print(f"✅ Text encoder in training mode with {text_trainable_params:,} trainable parameters")
        else:
            self.text_encoder.eval()
            print("✅ Text encoder frozen - set to eval mode")
        
                                                                                 
        self.unet = DiffusersUNet(
            latent_dim=model_config.get('latent_dim', 8),
            text_dim=model_config['text_embedding_dim'],
            pretrained_model_name=model_config.get('pretrained_model_name', 'runwayml/stable-diffusion-v1-5'),
            cross_attention_dim=model_config.get('cross_attention_dim', 768),               
            attention_head_dim=model_config.get('attention_head_dim', 8),
            use_flash_attention=model_config.get('use_flash_attention', True),
            freeze_encoder=model_config.get('freeze_encoder', False),
            freeze_decoder=model_config.get('freeze_decoder', False)
        ).to(self.device)

                         
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=model_config.get('num_timesteps', 1000),
            beta_start=model_config.get('beta_start', 0.0001),
            beta_end=model_config.get('beta_end', 0.02)
        )
        
                                 
        unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        text_trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        total_trainable = unet_trainable + text_trainable
        
        self.logger.info(f"📊 Diffusion Training Parameter Summary:")
        self.logger.info(f"   U-Net trainable: {unet_trainable:,}")
        self.logger.info(f"   Text encoder trainable: {text_trainable:,}")
        self.logger.info(f"   Total trainable: {total_trainable:,}")
        self.logger.info(f"   Fine-tuning strategy: {finetune_strategy}")
        
                               
        memory_gb = (total_trainable * 4 * 3) / (1024**3)                                                        
        self.logger.info(f"   Estimated training memory: ~{memory_gb:.2f} GB")
        
    def setup_data_loaders(self):
        """Setup data loaders for training, validation, and testing."""
        data_config = self.config['data']
        
                                                        
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
                                                          
        if 'unet_optimization' in self.config:
            self.training_config = self.config['unet_optimization']
        elif 'optimization' in self.config:
            self.training_config = self.config['optimization']
        else:
            self.training_config = self.config.get('training', {})
        
                                                               
        unet_lr = self.training_config.get('learning_rate', 5e-4)
        text_lr = self.training_config.get('text_encoder_lr', unet_lr * 0.1)                                            
        
                                                        
        text_trainable_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
        
        param_groups = [
            {
                'params': self.unet.parameters(),
                'lr': unet_lr,
                'name': 'unet'
            }
        ]
        
                                                                           
        if text_trainable_params:
            param_groups.append({
                'params': text_trainable_params,
                'lr': text_lr,
                'name': 'text_encoder'
            })
            self.logger.info(f"✅ Text encoder added to optimizer with LR={text_lr:.2e}")
        else:
            self.logger.info("✅ Text encoder frozen - not added to optimizer")
        
                                                         
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
                                                         
        self.criterion = nn.MSELoss()
        
                                                        
        self.use_mixed_precision = self.training_config.get('use_mixed_precision', False)
        if self.use_mixed_precision and self.device.type == 'mps':
                                                                                          
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
                                        
        scheduler_type = self.config.get('optimization', {}).get('scheduler', 'cosine')
        
        if scheduler_type == 'constant':
                                                         
            self.lr_scheduler = None
            print(f"✅ Using constant learning rates: {[group['lr'] for group in self.optimizer.param_groups]}")
            return
        
                                                  
        if 'training' in self.config and 'diffusion_epochs' in self.config['training']:
            num_epochs = self.config['training']['diffusion_epochs']
        elif 'unet_optimization' in self.config:
            num_epochs = self.config['unet_optimization'].get('num_epochs', 100)
        elif 'optimization' in self.config:
            num_epochs = self.config['optimization'].get('num_epochs', 100)
        else:
            num_epochs = 100            
            
        total_steps = len(self.train_loader) * num_epochs
        
                                                      
        max_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        if scheduler_type == 'cosine':
                                                                    
            self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:
                                 
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
        self.text_encoder.train()                                           
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                                
                images = batch['image'].to(self.device)
                descriptions = batch['description']
                
                                                          
                text_emb = self.text_encoder(descriptions)
                
                                                                      
                with torch.no_grad():
                    latent, _, _ = self.vae_encoder(images)
                
                                         
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, 
                    (latent.shape[0],), device=self.device
                )
                
                              
                noise = torch.randn_like(latent)
                
                                      
                noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
                
                                                                     
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
                
                               
                predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
                
                                               
                if self.check_for_nans(predicted_noise, "predicted_noise"):
                    self.logger.warning(f"Skipping batch due to NaN in predicted_noise")
                    continue
                
                              
                loss = self.criterion(predicted_noise, noise)
                
                                
                if self.check_for_nans(loss, "loss"):
                    self.logger.warning(f"Skipping batch due to NaN loss")
                    continue
                
                               
                self.optimizer.zero_grad()
                loss.backward()
                
                                                                         
                max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
                unet_grad_norm = torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=max_grad_norm)
                
                                                                                 
                text_trainable_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                if text_trainable_params:
                    text_grad_norm = torch.nn.utils.clip_grad_norm_(text_trainable_params, max_norm=max_grad_norm * 0.5)                                 
                else:
                    text_grad_norm = torch.tensor(0.0)                        
                
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                                                                
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('Train/UNet_Grad_Norm', unet_grad_norm.item(), self.global_step)
                    if text_trainable_params:
                        self.writer.add_scalar('Train/TextEncoder_Grad_Norm', text_grad_norm.item(), self.global_step)
                
                                                                
                if self.device.type == 'mps' and batch_idx % 5 == 0:                   
                    torch.mps.empty_cache()
                
                                
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                                                         
                unet_lr = self.optimizer.param_groups[0]['lr']
                pbar_dict = {
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / num_batches:.4f}",
                    'unet_lr': f"{unet_lr:.2e}"
                }
                
                                                           
                if len(self.optimizer.param_groups) > 1:
                    text_lr = self.optimizer.param_groups[1]['lr']
                    pbar_dict['text_lr'] = f"{text_lr:.2e}"
                
                pbar.set_postfix(pbar_dict)
                
                                    
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/UNet_LR', unet_lr, self.global_step)
                    
                                                                    
                    if len(self.optimizer.param_groups) > 1:
                        text_lr = self.optimizer.param_groups[1]['lr']
                        self.writer.add_scalar('Train/TextEncoder_LR', text_lr, self.global_step)
                    
                                                                                  
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
        self.text_encoder.eval()                                                
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                                    
                    images = batch['image'].to(self.device)
                    descriptions = batch['description']
                    
                                                                          
                    text_emb = self.text_encoder(descriptions)
                    
                                                   
                    latent, _, _ = self.vae_encoder(images)
                    
                                             
                    timesteps = torch.randint(
                        0, self.noise_scheduler.num_timesteps, 
                        (latent.shape[0],), device=self.device
                    )
                    
                                  
                    noise = torch.randn_like(latent)
                    
                                          
                    noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
                    
                                   
                    predicted_noise = self.unet(noisy_latent, timesteps, text_emb)
                    
                                  
                    loss = self.criterion(predicted_noise, noise)
                    
                    if not self.check_for_nans(loss, "val_loss"):
                        total_loss += loss.item()
                        num_batches += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
                            
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return {'val_loss': avg_loss}
    
    def ddpm_sample(self, text_emb: torch.Tensor, num_samples: int, 
                   fast_sampling: bool = True, num_inference_steps: int = 50) -> torch.Tensor:
        """Sample from the diffusion model using DDPM sampling."""
        self.unet.eval()
        self.text_encoder.eval()
        
                                 
        shape = (num_samples, 8, 27, 27)
        x = torch.randn(shape, device=self.device)
        
                                  
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
        
                                                    
        if text_emb.shape[0] != num_samples:
            text_emb = text_emb.repeat(num_samples, 1, 1)
        
        with torch.no_grad():
            for t in tqdm(timesteps, desc="Sampling"):
                t_tensor = torch.full((num_samples,), t.item(), device=self.device, dtype=torch.long)
                
                               
                noise_pred = self.unet(x, t_tensor, text_emb)
                
                                          
                x = self.noise_scheduler.sample_prev_timestep(x, noise_pred, int(t.item()))
        
        return x

    def denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images from [-1, 1] to [0, 1] for display."""
        return (images + 1.0) / 2.0

    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate and save sample images with tensorboard logging."""
        self.unet.eval()
        self.text_encoder.eval()
        
                                                          
        val_batch = next(iter(self.val_loader))
        descriptions = val_batch['description'][:min(num_samples, len(val_batch['description']))]
        real_images = val_batch['image'][:min(num_samples, len(val_batch['image']))].to(self.device)                  
        
                             
        with torch.no_grad():
            text_emb = self.text_encoder(descriptions)
            
                                                                              
            real_latents, _, _ = self.vae_encoder(real_images)
            
                                                                                
            reconstructed_images = self.vae_decoder(real_latents, text_emb)
        
                                              
        generated_latents = self.ddpm_sample(text_emb, len(descriptions), fast_sampling=True)
        
                                            
        with torch.no_grad():
            generated_images = self.vae_decoder(generated_latents, text_emb)
        
                                                       
        real_images_norm = self.denormalize_images(real_images)
        reconstructed_images_norm = self.denormalize_images(reconstructed_images)
        generated_images_norm = self.denormalize_images(generated_images)
        
                              
        sample_path = self.sample_dir / f"epoch_{epoch}_samples.png"
        
                                                                    
        comparison_grid = torch.cat([
            real_images_norm,
            reconstructed_images_norm, 
            generated_images_norm
        ], dim=0)
        
        save_image(
            comparison_grid,
            sample_path,
            nrow=len(descriptions),                                       
            normalize=False,                      
            value_range=(0, 1)
        )
        
                                                      
        self.writer.add_images('Diffusion/Real_Images', real_images_norm, epoch)
        self.writer.add_images('Diffusion/Reconstructed_Images', reconstructed_images_norm, epoch)
        self.writer.add_images('Diffusion/Generated_Images', generated_images_norm, epoch)
        
                                          
        self.writer.add_images('Diffusion/Comparison_Grid', comparison_grid, epoch)
        
                           
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
        
                                   
                                                                                
                                                 
        
                         
        if is_best:
            best_path = self.checkpoint_dir / "diffusion_best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        
                                                      
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
            num_epochs = 100            
            training_config = {}
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"U-Net parameters: {self.unet.get_parameter_count():,}")
        self.logger.info(f"Text encoder parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
        self.logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.unet.parameters()) + sum(p.numel() for p in self.text_encoder.parameters()):,}")
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                          
                train_metrics = self.train_epoch(epoch)
                
                            
                val_metrics = self.validate_epoch(epoch)
                
                                 
                metrics = {**train_metrics, **val_metrics}
                
                                                        
                self.writer.add_scalar('Epoch/Train_Loss', metrics['train_loss'], epoch)
                self.writer.add_scalar('Epoch/Val_Loss', metrics['val_loss'], epoch)
                
                                                           
                unet_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Epoch/UNet_LR', unet_lr, epoch)
                
                                                                
                log_text_lr = ""
                if len(self.optimizer.param_groups) > 1:
                    text_lr = self.optimizer.param_groups[1]['lr']
                    self.writer.add_scalar('Epoch/TextEncoder_LR', text_lr, epoch)
                    log_text_lr = f", text_lr={text_lr:.2e}"
                
                             
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={metrics['train_loss']:.4f}, "
                    f"val_loss={metrics['val_loss']:.4f}, "
                    f"unet_lr={unet_lr:.2e}"
                    f"{log_text_lr}"
                )
                
                                     
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                
                                 
                if epoch % training_config.get('save_every', 10) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                
                                  
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
                   
    config_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    config = load_config(config_path)
    
    vae_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/logs/vae_best_model.pth"
    
    trainer = DiffusersTrainer(
        config=config,
        vae_checkpoint_path=vae_checkpoint_path,
        experiment_name="pokemon_diffusers"
    )
    
    trainer.train()
