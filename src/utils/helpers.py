import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import yaml
import json
from datetime import datetime


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def tensor_to_image(tensor):
    """Convert tensor to PIL image."""
    # Denormalize from [-1, 1] to [0, 1]
    image = (tensor + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy
    if len(image.shape) == 4:
        image = image[0]  # Take first image from batch
    
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)


def save_image_grid(images, path, nrow=4, figsize=(12, 12)):
    """Save a grid of images."""
    batch_size = images.shape[0]
    ncol = min(nrow, batch_size)
    nrow = (batch_size + ncol - 1) // ncol
    
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    if nrow == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(batch_size):
        row = i // ncol
        col = i % ncol
        
        image = tensor_to_image(images[i])
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(batch_size, nrow * ncol):
        row = i // ncol
        col = i % ncol
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()


def save_attention_visualization(attention_weights, tokens, image_path, save_path):
    """Save attention weights visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = tensor_to_image(image_path)
    
    ax1.imshow(image)
    ax1.set_title('Generated Image')
    ax1.axis('off')
    
    # Display attention weights
    attention_weights = attention_weights.cpu().numpy()
    if len(attention_weights.shape) > 1:
        attention_weights = attention_weights[0]  # Take first in batch
    
    # Truncate tokens to match attention weights
    if len(tokens) > len(attention_weights):
        tokens = tokens[:len(attention_weights)]
    elif len(tokens) < len(attention_weights):
        attention_weights = attention_weights[:len(tokens)]
    
    # Create bar plot
    bars = ax2.bar(range(len(tokens)), attention_weights)
    ax2.set_xlabel('Tokens')
    ax2.set_ylabel('Attention Weight')
    ax2.set_title('Attention Weights')
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right')
    
    # Color bars based on attention weight
    max_weight = max(attention_weights)
    for bar, weight in zip(bars, attention_weights):
        bar.set_color(plt.cm.get_cmap("viridis")(weight / max_weight))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


class Logger:
    """Simple logger for training progress."""
    
    def __init__(self, log_dir, log_file='train.log'):
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_path = os.path.join(log_dir, log_file)
        
        # Initialize log file
        with open(self.log_path, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("=" * 50 + "\n")
    
    def log(self, message):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_path, 'a') as f:
            f.write(log_message + "\n")
    
    def log_metrics(self, epoch, metrics):
        """Log training metrics."""
        message = f"Epoch {epoch}: "
        for key, value in metrics.items():
            message += f"{key}: {value:.6f}, "
        message = message.rstrip(", ")
        
        self.log(message)


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(self, checkpoint_dir, keep_last=5):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.checkpoint_list = []
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_list.append(checkpoint_path)
        
        # Keep only last N checkpoints
        if len(self.checkpoint_list) > self.keep_last:
            old_checkpoint = self.checkpoint_list.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        return checkpoint_path
    
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def get_latest_checkpoint(self):
        """Get path to the latest checkpoint."""
        if not self.checkpoint_list:
            return None
        return self.checkpoint_list[-1]


def count_parameters(model):
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_shape=None):
    """Print model summary."""
    print("Model Summary:")
    print("=" * 50)
    
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    
    if input_shape is not None:
        print(f"Input shape: {input_shape}")
    
    print("\nModel architecture:")
    print(model)
    
    print("=" * 50)


def create_directories(config):
    """Create necessary directories for training."""
    directories = [
        config['training']['checkpoint_dir'],
        config['training']['log_dir'],
        config['data']['data_dir'],
        'outputs',
        'outputs/generated',
        'outputs/attention_maps'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_sample_outputs(model, tokenizer, sample_texts, device, output_dir, epoch):
    """Save sample generated outputs during training."""
    model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            # Generate image
            generated_images = model.generate([text], num_samples=1, return_attention=False)
            
            # Save image
            image_path = os.path.join(output_dir, f'epoch_{epoch:03d}_sample_{i}.png')
            image = tensor_to_image(generated_images[0])
            image.save(image_path)
    
    model.train()


def format_time(seconds):
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.0f}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:.0f}h {minutes:.0f}m {seconds:.1f}s"