import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import yaml
import time                  
from tqdm import tqdm                         

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
        
                      
        self.device = get_device()
        
                                 
        self.kl_anneal_start = config['training'].get('kl_anneal_start', 0)
        self.kl_anneal_end = config['training'].get('kl_anneal_end', 50)
        self.kl_weight_start = config['training'].get('kl_weight_start', 0.0)
        self.kl_weight_end = config['training'].get('kl_weight_end', 1.0)
        self.free_bits = config['training'].get('free_bits', 0.5)                            
        self.kl_annealing = config['training'].get('kl_annealing', True)                                  
        
                                
        self.current_epoch = 0
        self.global_step = 0
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
        
                                                                                          
        finetune_strategy = model_config.get('bert_finetune_strategy', 'minimal')
        
                                                 
        self.text_encoder = TextEncoder(
            model_name=model_config['bert_model'],
            hidden_dim=model_config['text_embedding_dim'],
            finetune_strategy=finetune_strategy
        ).to(self.device)
        
        print(f"Text encoder trainable parameters: {sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad):,}")
            
                   
        self.vae = PokemonVAE(
            latent_dim=model_config.get('latent_dim', 8),                                
            text_dim=model_config['text_embedding_dim']
        ).to(self.device)
        
        self.logger.info(f"VAE initialized with {sum(p.numel() for p in self.vae.parameters())} parameters")
        
                                              
        vae_trainable = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        text_trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        total_trainable = vae_trainable + text_trainable
        self.logger.info(f"Total trainable parameters: VAE={vae_trainable:,}, TextEncoder={text_trainable:,}, Total={total_trainable:,}")
        
                               
        memory_gb = (total_trainable * 4 * 3) / (1024**3)                                                        
        self.logger.info(f"Estimated training memory usage: ~{memory_gb:.2f} GB")
        
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
        
                                                              
        vae_lr = opt_config['learning_rate']
        text_lr = opt_config.get('text_encoder_lr', vae_lr * 0.1)                                         
        
        param_groups = [
            {
                'params': self.vae.parameters(),
                'lr': vae_lr,
                'name': 'vae'
            },
            {
                'params': self.text_encoder.parameters(), 
                'lr': text_lr,
                'name': 'text_encoder'
            }
        ]
        
                                                 
        if opt_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999))
            )
        elif opt_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        
        self.logger.info(f"Optimizer setup: VAE LR={vae_lr:.2e}, Text Encoder LR={text_lr:.2e}")
        
                                 
        scheduler_type = opt_config.get('scheduler', 'constant')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['vae_epochs']
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=opt_config.get('step_size', 30),
                gamma=opt_config.get('gamma', 0.1)
            )
        else:
                                                              
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, 
                lr_lambda=lambda epoch: 1.0
            )
        
        self.logger.info(f"Scheduler: {scheduler_type}")
        
                                            
        self.criterion = CombinedLoss(
            reconstruction_weight=self.config['training'].get('reconstruction_weight', 1.0),
            perceptual_weight=self.config['training'].get('perceptual_weight', 0.1),
            kl_weight=self.config['training'].get('kl_weight', 0.01)
        ).to(self.device)
        
    def setup_monitoring(self):
        """Setup monitoring and logging tools."""
                            
        self.tb_writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
    
    def get_kl_weight(self, epoch: int) -> float:
        """Get current KL weight with annealing."""
        if epoch < self.kl_anneal_start:
            return self.kl_weight_start
        elif epoch >= self.kl_anneal_end:
            return self.kl_weight_end
        else:
                              
            progress = (epoch - self.kl_anneal_start) / (self.kl_anneal_end - self.kl_anneal_start)
            return self.kl_weight_start + progress * (self.kl_weight_end - self.kl_weight_start)
    
    def compute_free_bits_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL loss with free bits to prevent posterior collapse."""
                                                                               
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
                                                     
        kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        
                                                     
        kl_loss = kl_per_dim.sum(dim=-1).mean()
        
        return kl_loss
    
    def compute_vae_loss(self, outputs: Dict[str, torch.Tensor], target_images: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VAE loss with perceptual loss, KL annealing and free bits."""
        reconstructed = outputs['reconstructed']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
                                             
        total_loss, loss_dict = self.criterion(reconstructed, target_images, mu, logvar)
        
                            
        kl_weight = self.get_kl_weight(epoch)
        
                                                                                    
        if self.kl_annealing:
                                                           
            recon_loss = self.criterion.l1_loss(reconstructed, target_images)
            perceptual_loss = self.criterion.perceptual_loss(
                (reconstructed + 1.0) / 2.0, (target_images + 1.0) / 2.0
            )
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
            
                                                          
            total_loss = (self.criterion.reconstruction_weight * recon_loss + 
                         self.criterion.perceptual_weight * perceptual_loss + 
                         kl_weight * kl_loss)
            
                                                   
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
        self.text_encoder.train()                                 
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
                                               
        epoch_start_time = time.time()
        batch_times = []
        
                                         
        pbar = tqdm(enumerate(self.data_loaders['train']), 
                   total=num_batches,
                   desc=f'Epoch {epoch+1}/{self.config["training"]["vae_epochs"]}',
                   leave=False,
                   dynamic_ncols=True)
        
        for batch_idx, batch in pbar:
            batch_start_time = time.time()
            
            images = batch['image'].to(self.device)
            descriptions = batch['full_description']
            
                                                              
            text_start_time = time.time()
            text_emb = self.text_encoder(descriptions)
            text_time = time.time() - text_start_time
            
                                      
            vae_start_time = time.time()
            outputs = self.vae(images, text_emb, mode='train')
            vae_time = time.time() - vae_start_time
            
                          
            loss_start_time = time.time()
            loss, loss_dict = self.compute_vae_loss(outputs, images, epoch)
            loss_time = time.time() - loss_start_time
            
                           
            backward_start_time = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            
                                                             
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=0.5)                               
            
            self.optimizer.step()
            backward_time = time.time() - backward_start_time
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
                            
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['reconstruction_loss']
            total_kl_loss += loss_dict['kl_loss']
            
                                                      
            avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)                   
            pbar.set_postfix({
                'Loss': f'{loss_dict["total_loss"]:.4f}',
                'Recon': f'{loss_dict["reconstruction_loss"]:.4f}',
                'KL': f'{loss_dict["kl_loss"]:.4f}',
                'KL_w': f'{loss_dict["kl_weight"]:.3f}',
                'Time': f'{avg_batch_time:.2f}s',
                'Text': f'{text_time:.3f}s',
                'VAE': f'{vae_time:.3f}s',
                'Loss_comp': f'{loss_time:.3f}s',
                'Backwd': f'{backward_time:.3f}s'
            })
            
                                                  
            if batch_idx % self.config['training']['log_every'] == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                               f'Loss: {loss_dict["total_loss"]:.4f}, '
                               f'Recon: {loss_dict["reconstruction_loss"]:.4f}, '
                               f'KL: {loss_dict["kl_loss"]:.4f}, '
                               f'KL Weight: {loss_dict["kl_weight"]:.4f}, '
                               f'Batch Time: {avg_batch_time:.2f}s '
                               f'(Text: {text_time:.3f}s, VAE: {vae_time:.3f}s, '
                               f'Loss: {loss_time:.3f}s, Backward: {backward_time:.3f}s)')
            
            self.global_step += 1
        
        pbar.close()
        
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
                                  
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        self.logger.info(f'Epoch {epoch} completed in {epoch_time:.1f}s, '
                        f'avg batch time: {avg_batch_time:.2f}s, '
                        f'estimated batches/hour: {3600/avg_batch_time:.1f}')
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'epoch_time': epoch_time,
            'avg_batch_time': avg_batch_time
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.vae.eval()
        self.text_encoder.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = len(self.data_loaders['val'])
        
                                            
        pbar = tqdm(self.data_loaders['val'], 
                   desc=f'Validation Epoch {epoch+1}',
                   leave=False,
                   dynamic_ncols=True)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                descriptions = batch['full_description']
                
                             
                text_emb = self.text_encoder(descriptions)
                
                                          
                outputs = self.vae(images, text_emb, mode='train')
                
                              
                loss, loss_dict = self.compute_vae_loss(outputs, images, epoch)
                
                                
                total_loss += loss_dict['total_loss']
                total_recon_loss += loss_dict['reconstruction_loss']
                total_kl_loss += loss_dict['kl_loss']
                
                                     
                pbar.set_postfix({
                    'Val_Loss': f'{loss_dict["total_loss"]:.4f}',
                    'Val_Recon': f'{loss_dict["reconstruction_loss"]:.4f}',
                    'Val_KL': f'{loss_dict["kl_loss"]:.4f}'
                })
        
        pbar.close()
        
                                  
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
        
                                        
        val_batch = next(iter(self.data_loaders['val']))
        batch_size = val_batch['image'].shape[0]
        num_samples = min(num_samples, batch_size)                                     
        descriptions = val_batch['full_description'][:num_samples]
        real_images = val_batch['image'][:num_samples]
        
        with torch.no_grad():
                         
            text_emb = self.text_encoder(descriptions)
            
                                      
            recon_outputs = self.vae(real_images.to(self.device), text_emb, mode='train')
            reconstructed = recon_outputs['reconstructed']
            
                                                     
            generated = self.vae.sample(num_samples, text_emb, self.device)
        
                                          
        real_images = self.denormalize_images(real_images)
        reconstructed = self.denormalize_images(reconstructed)
        generated = self.denormalize_images(generated)
        
                                
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
        
        for i in range(num_samples):
                        
            axes[0, i].imshow(real_images[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title(f'Real')
            axes[0, i].axis('off')
            
                                 
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title(f'Reconstructed')
            axes[1, i].axis('off')
            
                             
            axes[2, i].imshow(generated[i].permute(1, 2, 0).cpu().numpy())
            axes[2, i].set_title(f'Generated')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.sample_dir / f'vae_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
                            
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
        
                                   
                                                                                               
        
                         
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'vae_best_model.pth')
            self.logger.info(f'Saved best model at epoch {epoch}')
        
                                      
                                                                                    
                                  
                                
                                                     
                                         
    
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
        
                                        
        epoch_pbar = tqdm(range(self.current_epoch, self.config['training']['vae_epochs']),
                         desc='VAE Training',
                         unit='epoch',
                         dynamic_ncols=True)
        
        for epoch in epoch_pbar:
            epoch_desc = f'Epoch {epoch + 1}/{self.config["training"]["vae_epochs"]}'
            epoch_pbar.set_description(epoch_desc)
            
                   
            train_metrics = self.train_epoch(epoch)
            
                      
            val_metrics = self.validate_epoch(epoch)
            
                                  
            self.scheduler.step()
            
                         
            kl_weight = self.get_kl_weight(epoch)
            self.logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                           f'Recon: {train_metrics["recon_loss"]:.4f}, '
                           f'KL: {train_metrics["kl_loss"]:.4f}, '
                           f'KL Weight: {kl_weight:.4f}')
            self.logger.info(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                           f'Recon: {val_metrics["recon_loss"]:.4f}, '
                           f'KL: {val_metrics["kl_loss"]:.4f}')
            
                                                            
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_metrics["loss"]:.4f}',
                'Val_Loss': f'{val_metrics["loss"]:.4f}',
                'Epoch_Time': f'{train_metrics["epoch_time"]:.1f}s',
                'Batch_Time': f'{train_metrics["avg_batch_time"]:.2f}s'
            })
            
                                 
            self.tb_writer.add_scalar('VAE Train/Loss', train_metrics['loss'], epoch)
            self.tb_writer.add_scalar('VAE Train/Recon_Loss', train_metrics['recon_loss'], epoch)
            self.tb_writer.add_scalar('VAE Train/KL_Loss', train_metrics['kl_loss'], epoch)
            self.tb_writer.add_scalar('VAE Train/KL_Weight', kl_weight, epoch)
            self.tb_writer.add_scalar('VAE Train/Epoch_Time', train_metrics['epoch_time'], epoch)
            self.tb_writer.add_scalar('VAE Train/Batch_Time', train_metrics['avg_batch_time'], epoch)
            self.tb_writer.add_scalar('VAE Val/Loss', val_metrics['loss'], epoch)
            self.tb_writer.add_scalar('VAE Val/Recon_Loss', val_metrics['recon_loss'], epoch)
            self.tb_writer.add_scalar('VAE Val/KL_Loss', val_metrics['kl_loss'], epoch)
            
                              
            if epoch % self.config['training']['sample_every'] == 0:
                self.generate_samples(epoch)
            
                             
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            self.current_epoch = epoch + 1
        
        epoch_pbar.close()
        self.logger.info('VAE training completed!')
        self.tb_writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
                        
    config_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    config = load_config(config_path)
    
                        
    trainer = VAETrainer(config, experiment_name="pokemon_vae_stage1")
    
                    
    trainer.train()
