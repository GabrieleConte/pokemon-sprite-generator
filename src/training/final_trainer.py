import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import yaml

from src.models.vae_decoder import VAEEncoder, VAEDecoder
from src.models.text_encoder import TextEncoder
from src.models.unet import UNet
from src.models.clip_loss import CLIPLoss
from src.data import create_data_loaders
from src.utils import get_device


class NoiseScheduler:
    """Noise scheduler for diffusion sampling."""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
                                   
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
                                        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
                                          
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
                                        
        posterior_variance = self.betas * (1.0 - torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)
                                   
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
        
                               
        sqrt_recip_alpha = self.sqrt_recip_alphas[timestep]
        beta = self.betas[timestep]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep]
        
                      
        mean = sqrt_recip_alpha * (x_t - beta * predicted_noise / sqrt_one_minus_alpha_cumprod)
        
                                     
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
        
                                         
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        
                              
        self.vae_encoder = VAEEncoder(
            input_channels=3,
            latent_dim=text_encoder_config.get('latent_dim', 8)                                
        )
        
                                                                          
        self.vae_decoder = VAEDecoder(
            latent_dim=text_encoder_config.get('latent_dim', 8),                               
            text_dim=text_encoder_config['text_embedding_dim'],
            output_channels=3
        )
        
                          
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
        
                                         
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        
                                
        diffusion_checkpoint = torch.load(diffusion_path, map_location='cpu')
        
        self.unet = UNet(
            latent_dim=text_encoder_config.get('latent_dim', 8),                               
            text_dim=text_encoder_config['text_embedding_dim'],
            time_emb_dim=text_encoder_config.get('time_emb_dim', 128),
            num_heads=text_encoder_config.get('num_heads', 8)
        )
        
        self.unet.load_state_dict(diffusion_checkpoint['unet_state_dict'])
        
                                
        for param in self.unet.parameters():
            param.requires_grad = False
        
                                           
        self.text_encoder = TextEncoder(
            model_name=text_encoder_config['bert_model'],
            hidden_dim=text_encoder_config['text_embedding_dim']
        )
        
                                                
        if 'text_encoder_state_dict' in vae_checkpoint:
            self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'])
        
                              
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=text_encoder_config.get('num_timesteps', 1000),
            beta_start=text_encoder_config.get('beta_start', 0.0001),
            beta_end=text_encoder_config.get('beta_end', 0.02)
        )
        
                          
        self.latent_dim = text_encoder_config.get('latent_dim', 8)                               
        
    def forward(self, text_list: List[str], num_inference_steps: int = 50, mode: str = 'generate') -> torch.Tensor:
        """
        Generate Pokemon images from text descriptions using diffusion.
        
        Args:
            text_list: List of text descriptions
            num_inference_steps: Number of diffusion steps for sampling
            mode: 'generate' for pure generation, 'reconstruct' for image reconstruction
            
        Returns:
            Generated images [batch_size, 3, 215, 215]
        """
                     
        text_emb = self.text_encoder(text_list)
        
        if mode == 'generate':
                                                   
            batch_size = text_emb.size(0)
            latent = torch.randn(batch_size, self.latent_dim, 27, 27, device=text_emb.device)                         
            
                                
            step_size = max(1, self.noise_scheduler.num_timesteps // num_inference_steps)
            
            for i in range(num_inference_steps):
                timestep = self.noise_scheduler.num_timesteps - 1 - i * step_size
                timestep = max(0, timestep)
                
                                        
                timesteps = torch.full((batch_size,), timestep, device=text_emb.device, dtype=torch.long)
                
                                          
                with torch.no_grad():
                    predicted_noise = self.unet(latent, timesteps, text_emb)
                
                                          
                if timestep > 0:
                    latent = self.noise_scheduler.sample_previous_timestep(latent, predicted_noise, timestep)
                else:
                                                              
                    latent = latent - predicted_noise
        
        else:                                      
                                                                      
            raise NotImplementedError("Reconstruction mode requires input images")
        
                                                  
        generated = self.vae_decoder(latent, text_emb)
        
        return generated
    
    def encode_and_decode(self, images: torch.Tensor, text_list: List[str]) -> torch.Tensor:
        """
        Encode images to latent space and decode back (for training).
        
        Args:
            images: Input images [batch_size, 3, 215, 215]
            text_list: List of text descriptions
            
        Returns:
            Reconstructed images [batch_size, 3, 215, 215]
        """
                     
        text_emb = self.text_encoder(text_list)
        
                                       
        with torch.no_grad():
            latent, _, _ = self.vae_encoder(images)
        
                                                  
        reconstructed = self.vae_decoder(latent, text_emb)
        
        return reconstructed
    
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
        
                      
        self.device = get_device()
        print(f"Using device: {self.device}")
        
                           
        self.setup_directories()
        
                       
        self.setup_logging()
        
                     
        self.setup_model()
        
                            
        self.setup_data_loaders()
        
                            
        self.setup_optimization()
        
                          
        self.setup_monitoring()
        
                        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_phase = 'text_encoder'                             
        
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
        
                                
        self.generator = FinalPokemonGenerator(
            vae_path=self.vae_checkpoint_path,
            diffusion_path=self.diffusion_checkpoint_path,
            text_encoder_config=model_config
        ).to(self.device)
        
                                            
        self.clip_loss = CLIPLoss(
            device=str(self.device)
        ).to(self.device)
        
        self.logger.info(f"Final generator initialized")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in self.generator.parameters())}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}")
        self.logger.info(f"CLIP loss parameters (frozen): {sum(p.numel() for p in self.clip_loss.parameters())}")
        
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
        
                                                           
        param_groups = [
            {
                'params': self.generator.text_encoder.parameters(),
                'lr': opt_config['text_encoder_lr'],
                'name': 'text_encoder'
            }
        ]
        
                   
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
        
                       
        self.criterion = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def setup_monitoring(self):
        """Setup monitoring and logging tools."""
                            
        self.tb_writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
    
    def compute_generation_loss(self, generated_images: torch.Tensor, target_images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute generation loss."""
                                  
        l1_loss = self.l1_loss(generated_images, target_images)
        mse_loss = self.criterion(generated_images, target_images)
        
                              
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
                                                   
        self.clip_loss.eval()
        
        total_loss = 0.0
        total_l1_loss = 0.0
        total_mse_loss = 0.0
        total_clip_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        for batch_idx, batch in enumerate(self.data_loaders['train']):
            images = batch['image'].to(self.device)
            descriptions = batch['full_description']
            
                                                                                
            reconstructed_images = self.generator.encode_and_decode(images, descriptions)
            
                                     
            gen_loss, gen_loss_dict = self.compute_generation_loss(reconstructed_images, images)
            
                                         
            clip_loss = self.clip_loss(reconstructed_images, descriptions)
            
                           
            clip_weight = self.config['training'].get('clip_weight', 0.1)
            total_loss_value = gen_loss + clip_weight * clip_loss
            
                           
            self.optimizer.zero_grad()
            total_loss_value.backward()
            
                                                               
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
                            
            total_loss += total_loss_value.item()
            total_l1_loss += gen_loss_dict['l1_loss']
            total_mse_loss += gen_loss_dict['mse_loss']
            total_clip_loss += clip_loss.item()
            
                          
            if batch_idx % self.config['training']['log_every'] == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                               f'Phase: {self.training_phase}, '
                               f'Total: {total_loss_value.item():.4f}, '
                               f'Gen: {gen_loss.item():.4f}, '
                               f'CLIP: {clip_loss.item():.4f}')
            
            self.global_step += 1
        
                                  
        avg_loss = total_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_clip_loss = total_clip_loss / num_batches
        
        return {
            'loss': avg_loss,
            'l1_loss': avg_l1_loss,
            'mse_loss': avg_mse_loss,
            'clip_loss': avg_clip_loss
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
                
                                                                     
                reconstructed_images = self.generator.encode_and_decode(images, descriptions)
                
                              
                loss, loss_dict = self.compute_generation_loss(reconstructed_images, images)
                
                                
                total_loss += loss_dict['total_loss']
                total_l1_loss += loss_dict['l1_loss']
                total_mse_loss += loss_dict['mse_loss']
        
                                  
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
        
                                        
        val_batch = next(iter(self.data_loaders['val']))
        descriptions = val_batch['full_description'][:num_samples]
        real_images = val_batch['image'][:num_samples]
        
        with torch.no_grad():
            generated_images = self.generator(descriptions)
        
                                          
        real_images = self.denormalize_images(real_images)
        generated_images = self.denormalize_images(generated_images)
        
                                
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        for i in range(num_samples):
                        
            axes[0, i].imshow(real_images[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title('Real')
            axes[0, i].axis('off')
            
                             
            axes[1, i].imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title('Generated')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.sample_dir / f'final_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
                            
        self.tb_writer.add_images('Real_Images', real_images, epoch)
        self.tb_writer.add_images('Generated_Images', generated_images, epoch)
    
    def switch_to_joint_training(self):
        """Switch to joint training mode - unfreeze all parameters."""
        self.logger.info("Switching to joint training mode - unfreezing all parameters")
        
                                 
        self.generator.unfreeze_vae_decoder()
        self.generator.unfreeze_unet()
        self.training_phase = 'joint'
        
                                                  
        opt_config = self.config['optimization']
        param_groups = [
            {
                'params': self.generator.text_encoder.parameters(),
                'lr': opt_config['text_encoder_lr'],
                'name': 'text_encoder'
            },
            {
                'params': self.generator.vae_decoder.parameters(),
                'lr': opt_config.get('vae_decoder_lr', opt_config['text_encoder_lr'] * 0.1),
                'name': 'vae_decoder'
            },
            {
                'params': self.generator.unet.parameters(),
                'lr': opt_config.get('unet_lr', opt_config['text_encoder_lr'] * 0.1),
                'name': 'unet'
            }
        ]
        
                              
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
        
                                   
                                                                                                 
        
                         
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'final_best_model.pth')
            self.logger.info(f'Saved best model at epoch {epoch}')
        
                                        
                                                                                      
                                  
                                
                                                     
                                         
    
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
        
                                          
        phase1_epochs = self.config['training'].get('phase1_epochs', self.config['training']['final_epochs'] // 2)
        
        for epoch in range(self.current_epoch, self.config['training']['final_epochs']):
            self.logger.info(f'Epoch {epoch + 1}/{self.config["training"]["final_epochs"]}')
            
                                                         
            if epoch == phase1_epochs and self.training_phase == 'text_encoder':
                self.switch_to_joint_training()
            
                   
            train_metrics = self.train_epoch(epoch)
            
                      
            val_metrics = self.validate_epoch(epoch)
            
                                  
            self.scheduler.step()
            
                         
            self.logger.info(f'Phase: {self.training_phase}')
            self.logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                           f'L1: {train_metrics["l1_loss"]:.4f}, '
                           f'MSE: {train_metrics["mse_loss"]:.4f}, '
                           f'CLIP: {train_metrics["clip_loss"]:.4f}')
            self.logger.info(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                           f'L1: {val_metrics["l1_loss"]:.4f}, '
                           f'MSE: {val_metrics["mse_loss"]:.4f}')
            
                                 
            self.tb_writer.add_scalar('Final Train/Loss', train_metrics['loss'], epoch)
            self.tb_writer.add_scalar('Final Train/L1_Loss', train_metrics['l1_loss'], epoch)
            self.tb_writer.add_scalar('Final Train/MSE_Loss', train_metrics['mse_loss'], epoch)
            self.tb_writer.add_scalar('Final Train/CLIP_Loss', train_metrics['clip_loss'], epoch)
            self.tb_writer.add_scalar('Final Val/Loss', val_metrics['loss'], epoch)
            self.tb_writer.add_scalar('Final Val/L1_Loss', val_metrics['l1_loss'], epoch)
            self.tb_writer.add_scalar('Final Val/MSE_Loss', val_metrics['mse_loss'], epoch)
            
                              
            if epoch % self.config['training']['sample_every'] == 0:
                self.generate_samples(epoch)
            
                             
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
                        
    config_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    config = load_config(config_path)
    
                                   
    vae_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/experiments/pokemon_vae_stage1/checkpoints/vae_best_model.pth"
    
                                         
    diffusion_checkpoint_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/experiments/pokemon_diffusion_stage2/checkpoints/diffusion_best_model.pth"
    
                          
    trainer = FinalTrainer(
        config=config,
        vae_checkpoint_path=vae_checkpoint_path,
        diffusion_checkpoint_path=diffusion_checkpoint_path,
        experiment_name="pokemon_final_stage3"
    )
    
                    
    trainer.train()
