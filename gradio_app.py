#!/usr/bin/env python3
"""
Gradio app for Pokemon sprite generation using the 3-stage trained model.
Supports both text-only generation and image-conditioned generation.
"""

import os
import sys
import torch
import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union, List
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Suppress warnings for cleaner interface
warnings.filterwarnings("ignore")

from src.models.vae_decoder import VAEEncoder, VAEDecoder, PokemonVAE
from src.models.text_encoder import TextEncoder
from src.models.diffusers_unet import DiffusersUNet
from src.models.unet import UNet
from src.training.vae_trainer import load_config
from src.utils import get_device


class PokemonGradioGenerator:
    """
    Pokemon sprite generator with Gradio interface.
    Supports both text-only and image-conditioned generation.
    """
    
    def __init__(self, 
                 vae_checkpoint_path: str,
                 diffusion_checkpoint_path: str, 
                 config_path: str,
                 use_diffusers: bool = True):
        """
        Initialize the Pokemon generator for Gradio.
        
        Args:
            vae_checkpoint_path: Path to VAE checkpoint
            diffusion_checkpoint_path: Path to diffusion checkpoint
            config_path: Path to training configuration
            use_diffusers: Whether to use diffusers U-Net or custom U-Net
        """
        self.device = get_device()
        self.use_diffusers = use_diffusers
        print(f"🚀 Initializing Pokemon Generator on device: {self.device}")
        
        # Load configuration
        self.config = load_config(config_path)
        model_config = self.config['model']
        
        print("📋 Loading models...")
        
        # Initialize text encoder
        self.text_encoder = TextEncoder(
            model_name=model_config.get('bert_model', 'google-bert/bert-base-uncased'),
            hidden_dim=model_config.get('text_embedding_dim', 768),
            finetune_strategy=model_config.get('bert_finetune_strategy', 'minimal')
        ).to(self.device)
        
        # Initialize VAE components
        latent_dim = model_config.get('latent_dim', 8)
        text_dim = model_config.get('text_embedding_dim', 768)
        
        self.vae = PokemonVAE(latent_dim=latent_dim, text_dim=text_dim).to(self.device)
        
        # Initialize U-Net
        if use_diffusers:
            self.unet = DiffusersUNet(
                latent_dim=latent_dim,
                text_dim=text_dim,
                pretrained_model_name=model_config.get('pretrained_model_name', 'runwayml/stable-diffusion-v1-5'),
                cross_attention_dim=model_config.get('cross_attention_dim', 768),
                use_flash_attention=model_config.get('use_flash_attention', True)
            ).to(self.device)
        else:
            # Use custom U-Net if available
            try:
                self.unet = UNet(
                    latent_dim=latent_dim,
                    text_dim=text_dim,
                    time_emb_dim=model_config.get('time_emb_dim', 128),
                    num_heads=model_config.get('num_heads', 8)
                ).to(self.device)
            except Exception as e:
                print(f"⚠️ Custom U-Net failed, falling back to diffusers: {e}")
                self.unet = DiffusersUNet(
                    latent_dim=latent_dim,
                    text_dim=text_dim,
                    pretrained_model_name=model_config.get('pretrained_model_name', 'runwayml/stable-diffusion-v1-5'),
                    cross_attention_dim=model_config.get('cross_attention_dim', 768),
                    use_flash_attention=model_config.get('use_flash_attention', True)
                ).to(self.device)
                self.use_diffusers = True
        
        # Load model weights
        self.load_checkpoints(vae_checkpoint_path, diffusion_checkpoint_path)
        
        # Set models to evaluation mode
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        
        # Noise scheduler parameters
        self.num_timesteps = model_config.get('num_timesteps', 1000)
        self.beta_start = model_config.get('beta_start', 0.0001)
        self.beta_end = model_config.get('beta_end', 0.02)
        
        # Initialize noise scheduler
        self.setup_noise_scheduler()
        
        print("✅ Pokemon Generator ready!")
        
    def load_checkpoints(self, vae_path: str, diffusion_path: str):
        """Load model weights from checkpoints."""
        print(f"📂 Loading VAE checkpoint: {vae_path}")
        vae_checkpoint = torch.load(vae_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'vae_state_dict' in vae_checkpoint:
            self.vae.load_state_dict(vae_checkpoint['vae_state_dict'])
        elif 'model_state_dict' in vae_checkpoint:
            self.vae.load_state_dict(vae_checkpoint['model_state_dict'])
        else:
            self.vae.load_state_dict(vae_checkpoint)
            
        # Load text encoder if available in checkpoint
        if 'text_encoder_state_dict' in vae_checkpoint:
            self.text_encoder.load_state_dict(vae_checkpoint['text_encoder_state_dict'])
        
        print(f"📂 Loading U-Net checkpoint: {diffusion_path}")
        diffusion_checkpoint = torch.load(diffusion_path, map_location=self.device)
        
        # Handle different checkpoint formats for U-Net
        if 'unet_state_dict' in diffusion_checkpoint:
            self.unet.load_state_dict(diffusion_checkpoint['unet_state_dict'])
        elif 'model_state_dict' in diffusion_checkpoint:
            self.unet.load_state_dict(diffusion_checkpoint['model_state_dict'])
        else:
            self.unet.load_state_dict(diffusion_checkpoint)
            
    def setup_noise_scheduler(self):
        """Setup noise scheduler for diffusion sampling."""
        # Linear schedule for betas
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        
    def ddpm_sample(self, text_emb: torch.Tensor, num_inference_steps: int = 50, 
                   initial_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the diffusion model using DDPM.
        
        Args:
            text_emb: Text embeddings [batch_size, seq_len, text_dim]
            num_inference_steps: Number of denoising steps
            initial_latent: Optional initial latent (for image conditioning)
            
        Returns:
            Denoised latent representation
        """
        batch_size = text_emb.shape[0]
        
        if initial_latent is None:
            # Start from pure noise (text-only generation)
            latent = torch.randn(
                batch_size, 
                self.config['model'].get('latent_dim', 8), 
                27, 27,  # Based on VAE architecture
                device=self.device
            )
        else:
            # Start from provided latent (image-conditioned generation)
            latent = initial_latent.clone()
        
        # Sampling timesteps
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device
        )
        
        # DDPM sampling loop
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full((batch_size,), t.item(), dtype=torch.long, device=self.device)
            
            # Predict noise
            with torch.no_grad():
                if self.use_diffusers:
                    # Diffusers U-Net format
                    predicted_noise = self.unet(latent, t_tensor, text_emb)
                else:
                    # Custom U-Net format
                    predicted_noise = self.unet(latent, t_tensor, text_emb)
            
            # Compute denoising step
            if i < len(timesteps) - 1:
                # Not the last step
                next_t = timesteps[i + 1]
                alpha_t = self.alphas[t]
                alpha_t_next = self.alphas[next_t]
                
                # Compute denoised latent
                latent = (latent - (1 - alpha_t) / torch.sqrt(1 - self.alphas_cumprod[t]) * predicted_noise) / torch.sqrt(alpha_t)
                
                # Add noise for next step
                if next_t > 0:
                    noise = torch.randn_like(latent)
                    latent = torch.sqrt(alpha_t_next) * latent + torch.sqrt(1 - alpha_t_next) * noise
            else:
                # Last step - final denoising
                latent = (latent - (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_cumprod[t]) * predicted_noise) / torch.sqrt(self.alphas[t])
        
        return latent
    
    def generate_from_text(self, description: str, num_inference_steps: int = 50, 
                          seed: Optional[int] = None) -> Image.Image:
        """
        Generate Pokemon sprite from text description only.
        
        Args:
            description: Text description of the Pokemon
            num_inference_steps: Number of diffusion denoising steps
            seed: Random seed for reproducibility
            
        Returns:
            Generated Pokemon sprite as PIL Image
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        with torch.no_grad():
            # Encode text
            text_emb = self.text_encoder([description])
            
            # Sample from diffusion model (pure noise start)
            latent = self.ddpm_sample(text_emb, num_inference_steps)
            
            # Decode to image
            generated_image = self.vae.decode(latent, text_emb)
            
            # Convert to PIL Image
            return self.tensor_to_pil(generated_image[0])
    
    def generate_from_image_and_text(self, input_image: Image.Image, description: str, 
                                   num_inference_steps: int = 50, noise_strength: float = 0.7,
                                   seed: Optional[int] = None) -> Image.Image:
        """
        Generate Pokemon sprite from both image and text conditioning.
        
        Args:
            input_image: Input image to condition on
            description: Text description of desired modifications
            num_inference_steps: Number of diffusion denoising steps
            noise_strength: How much noise to add (0.0 = no change, 1.0 = pure noise)
            seed: Random seed for reproducibility
            
        Returns:
            Generated Pokemon sprite as PIL Image
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        with torch.no_grad():
            # Preprocess input image
            input_tensor = self.pil_to_tensor(input_image)
            
            # Encode image to latent space
            latent, _, _ = self.vae.encode(input_tensor.unsqueeze(0))
            
            # Add noise based on noise_strength
            if noise_strength > 0:
                noise = torch.randn_like(latent)
                # Mix original latent with noise
                latent = latent * (1 - noise_strength) + noise * noise_strength
            
            # Encode text
            text_emb = self.text_encoder([description])
            
            # Sample from diffusion model starting from noisy latent
            denoised_latent = self.ddpm_sample(text_emb, num_inference_steps, initial_latent=latent)
            
            # Decode to image
            generated_image = self.vae.decode(denoised_latent, text_emb)
            
            # Convert to PIL Image
            return self.tensor_to_pil(generated_image[0])
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        # Resize to expected size
        image = image.resize((215, 215), Image.Resampling.LANCZOS)
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        tensor = (tensor - 0.5) * 2  # [0, 1] to [-1, 1]
        
        return tensor.to(self.device)
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Move to CPU and denormalize
        tensor = tensor.cpu()
        tensor = (tensor + 1.0) / 2.0  # [-1, 1] to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and PIL
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)


def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    # Define paths (you may need to adjust these)
    VAE_CHECKPOINT = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/logs/v9/vae_best_model.pth"
    DIFFUSION_CHECKPOINT = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/logs/v9/diffusion_best_model.pth"
    CONFIG_PATH = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/config/train_config.yaml"
    
    # Check if checkpoints exist
    missing_files = []
    if not os.path.exists(VAE_CHECKPOINT):
        missing_files.append(f"VAE checkpoint: {VAE_CHECKPOINT}")
    if not os.path.exists(DIFFUSION_CHECKPOINT):
        missing_files.append(f"Diffusion checkpoint: {DIFFUSION_CHECKPOINT}")
    if not os.path.exists(CONFIG_PATH):
        missing_files.append(f"Config file: {CONFIG_PATH}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all model checkpoints are available.")
        return None
    
    # Initialize generator
    print("🔄 Initializing Pokemon Generator...")
    try:
        generator = PokemonGradioGenerator(
            vae_checkpoint_path=VAE_CHECKPOINT,
            diffusion_checkpoint_path=DIFFUSION_CHECKPOINT,
            config_path=CONFIG_PATH,
            use_diffusers=True  # Try diffusers first, fallback to custom if needed
        )
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return None
    
    # Define generation functions for Gradio
    def generate_text_only(description, steps, seed):
        """Wrapper for text-only generation."""
        try:
            if not description.strip():
                return None, "Please provide a description."
            
            image = generator.generate_from_text(
                description=description,
                num_inference_steps=int(steps),
                seed=int(seed) if seed else None
            )
            return image, f"✅ Generated Pokemon: '{description}'"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def generate_image_conditioned(input_image, description, steps, noise_strength, seed):
        """Wrapper for image-conditioned generation."""
        try:
            if input_image is None:
                return None, "Please provide an input image."
            if not description.strip():
                return None, "Please provide a description."
            
            image = generator.generate_from_image_and_text(
                input_image=input_image,
                description=description,
                num_inference_steps=int(steps),
                noise_strength=noise_strength,
                seed=int(seed) if seed else None
            )
            return image, f"✅ Generated conditioned Pokemon: '{description}'"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="Pokemon Sprite Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎮 Pokemon Sprite Generator
        
        Generate unique Pokemon sprites using AI! Choose between two modes:
        - **Text-Only**: Generate from scratch using only text descriptions
        - **Image + Text**: Modify existing images with text guidance
        """)
        
        with gr.Tabs():
            # Text-only generation tab
            with gr.TabItem("🎨 Text-Only Generation"):
                gr.Markdown("Generate Pokemon sprites from text descriptions only (pure diffusion)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Pokemon Description",
                            placeholder="A fire-type Pokemon with orange flames and wings...",
                            lines=3,
                            value="A fire-type Pokemon with orange flames and wings"
                        )
                        
                        text_steps = gr.Slider(
                            minimum=10, maximum=100, value=50, step=5,
                            label="Inference Steps (more = higher quality, slower)"
                        )
                        
                        text_seed = gr.Number(
                            label="Seed (optional, for reproducibility)",
                            value=None,
                            precision=0
                        )
                        
                        text_generate_btn = gr.Button("🎲 Generate Pokemon", variant="primary")
                    
                    with gr.Column(scale=1):
                        text_output_image = gr.Image(label="Generated Pokemon", type="pil")
                        text_output_text = gr.Textbox(label="Status", interactive=False)
                
                # Examples for text-only generation
                gr.Examples(
                    examples=[
                        ["A fire-type Pokemon with orange flames and wings", 50, 42],
                        ["A water-type Pokemon with blue scales and fins", 50, 123],
                        ["An electric-type Pokemon with yellow fur and lightning bolts", 50, 456],
                        ["A grass-type Pokemon with green leaves and flower petals", 50, 789],
                        ["A psychic-type Pokemon with purple aura and mystic powers", 50, 101],
                        ["A rock-type Pokemon with gray stone armor and crystal spikes", 50, 202],
                    ],
                    inputs=[text_input, text_steps, text_seed],
                    outputs=[text_output_image, text_output_text],
                    fn=generate_text_only,
                    cache_examples=False
                )
            
            # Image-conditioned generation tab
            with gr.TabItem("🖼️ Image + Text Conditioning"):
                gr.Markdown("Modify existing Pokemon images using text guidance")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Input Pokemon Image",
                            type="pil",
                            image_mode="RGB"
                        )
                        
                        image_text_input = gr.Textbox(
                            label="Modification Description",
                            placeholder="Make it more fiery with red flames...",
                            lines=3,
                            value="Make it more fiery with red and orange flames"
                        )
                        
                        image_steps = gr.Slider(
                            minimum=10, maximum=100, value=50, step=5,
                            label="Inference Steps"
                        )
                        
                        noise_strength = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                            label="Noise Strength (0.0 = no change, 1.0 = complete regeneration)"
                        )
                        
                        image_seed = gr.Number(
                            label="Seed (optional)",
                            value=None,
                            precision=0
                        )
                        
                        image_generate_btn = gr.Button("🎨 Transform Pokemon", variant="primary")
                    
                    with gr.Column(scale=1):
                        image_output_image = gr.Image(label="Transformed Pokemon", type="pil")
                        image_output_text = gr.Textbox(label="Status", interactive=False)
        
        # Connect buttons to functions
        text_generate_btn.click(
            fn=generate_text_only,
            inputs=[text_input, text_steps, text_seed],
            outputs=[text_output_image, text_output_text]
        )
        
        image_generate_btn.click(
            fn=generate_image_conditioned,
            inputs=[image_input, image_text_input, image_steps, noise_strength, image_seed],
            outputs=[image_output_image, image_output_text]
        )
        
        gr.Markdown("""
        ## 💡 Tips:
        - **Text-only**: Describe Pokemon type, colors, features (e.g., "fire-type", "wings", "flames")
        - **Image + Text**: Upload a Pokemon image and describe desired changes
        - **Inference Steps**: 20-30 for quick results, 50+ for better quality
        - **Noise Strength**: Low (0.1-0.3) for subtle changes, High (0.7-0.9) for major transformations
        - **Seed**: Use the same seed to reproduce results
        """)
    
    return interface


def main():
    """Main function to launch the Gradio app."""
    print("🚀 Starting Pokemon Sprite Generator Gradio App")
    
    # Create interface
    interface = create_gradio_interface()
    
    if interface is None:
        print("❌ Failed to create interface. Please check model checkpoints.")
        return
    
    # Launch the app
    print("🌐 Launching Gradio interface...")
    interface.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
