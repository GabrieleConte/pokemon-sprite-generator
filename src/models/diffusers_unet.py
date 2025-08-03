"""
Diffusers-based U-Net implementation for Pokemon sprite generation.
Uses a pre-trained Stable Diffusion U-Net and fine-tunes it for our specific requirements.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0


class DiffusersUNet(nn.Module):
    """
    Wrapper around pre-trained Stable Diffusion UNet2DConditionModel for Pokemon sprite generation.
    
    This model:
    - Loads pre-trained Stable Diffusion weights for better initialization
    - Adapts the model for our latent space [batch_size, 8, 27, 27]
    - Uses text conditioning with proper embedding dimension matching
    - Fine-tunes the pre-trained model rather than training from scratch
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        text_dim: int = 256,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        cross_attention_dim: int = 768,                                   
        attention_head_dim: int = 8,
        use_flash_attention: bool = True,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False
    ):
        """
        Initialize the pre-trained diffusers-based U-Net.
        
        Args:
            latent_dim: Number of input/output channels (8 for our VAE)
            text_dim: Dimension of our text embeddings (256 from our TextEncoder)
            pretrained_model_name: Name of the pre-trained Stable Diffusion model
            cross_attention_dim: Cross-attention dimension from pre-trained model (768 for SD 1.5)
            attention_head_dim: Dimension per attention head
            use_flash_attention: Whether to use flash attention for better performance
            freeze_encoder: Whether to freeze the encoder part during training
            freeze_decoder: Whether to freeze the decoder part during training
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.cross_attention_dim = cross_attention_dim
        self.pretrained_model_name = pretrained_model_name
        
        print(f"Loading pre-trained U-Net from {pretrained_model_name}...")
        
                                    
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.float32
        )
        
        print(f"✅ Loaded pre-trained U-Net with {sum(p.numel() for p in self.unet.parameters()):,} parameters")
        
                                               
        original_in_channels = getattr(self.unet.config, 'in_channels', 4)                   
        original_out_channels = getattr(self.unet.config, 'out_channels', 4)                   
        
        if original_in_channels != latent_dim:
            print(f"Adapting input channels: {original_in_channels} -> {latent_dim}")
            self._adapt_input_channels(original_in_channels, latent_dim)
        
        if original_out_channels != latent_dim:
            print(f"Adapting output channels: {original_out_channels} -> {latent_dim}")
            self._adapt_output_channels(original_out_channels, latent_dim)
        
                                             
                                                                                                    
        if text_dim != cross_attention_dim:
            print(f"Adding text projection layer: {text_dim} -> {cross_attention_dim}")
            self.text_projection = nn.Linear(text_dim, cross_attention_dim)
                                                                                                 
            nn.init.xavier_uniform_(self.text_projection.weight, gain=0.02)                                      
            nn.init.zeros_(self.text_projection.bias)
            
                                                              
            self.text_layer_norm = nn.LayerNorm(cross_attention_dim, eps=1e-6)
        else:
            print("Text dimensions match - no projection needed")
            self.text_projection = nn.Identity()
            self.text_layer_norm = nn.Identity()
        
                                             
        if use_flash_attention:
            self._setup_flash_attention()
        
                                                              
        if freeze_encoder and freeze_decoder:
            self.configure_training_mode("cross_attention_only")
        elif freeze_encoder and not freeze_decoder:
            self.configure_training_mode("decoder_only")
        else:
            self.configure_training_mode("full")
        
                                         
        self.print_training_summary()
        
    def _adapt_input_channels(self, original_channels: int, target_channels: int):
        """Adapt the input convolution to handle different channel counts."""
        original_conv = self.unet.conv_in
        
                                                        
        kernel_size = original_conv.kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            kernel_size = (kernel_size[0], kernel_size[1])
            
        stride = original_conv.stride
        if isinstance(stride, int):
            stride = (stride, stride)
        else:
            stride = (stride[0], stride[1])
            
        padding = original_conv.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        elif isinstance(padding, tuple) and len(padding) >= 2:
            padding = (padding[0], padding[1])
        elif isinstance(padding, tuple) and len(padding) == 1:
            padding = (padding[0], padding[0])
        elif isinstance(padding, str):
                                                                     
            pass
        else:
                              
            padding = (1, 1)
        
                                                     
        new_conv = nn.Conv2d(
            target_channels,
            original_conv.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=original_conv.bias is not None
        )
        
                                                                                 
        with torch.no_grad():
            if target_channels <= original_channels:
                                        
                new_conv.weight.data = original_conv.weight.data[:, :target_channels, :, :]
            else:
                                               
                repeat_factor = target_channels // original_channels
                remainder = target_channels % original_channels
                
                weight_parts = []
                for i in range(repeat_factor):
                    weight_parts.append(original_conv.weight.data)
                if remainder > 0:
                    weight_parts.append(original_conv.weight.data[:, :remainder, :, :])
                
                new_conv.weight.data = torch.cat(weight_parts, dim=1) / repeat_factor
            
            if new_conv.bias is not None and original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()
        
                           
        self.unet.conv_in = new_conv
        
    def _adapt_output_channels(self, original_channels: int, target_channels: int):
        """Adapt the output convolution to handle different channel counts."""
        original_conv = self.unet.conv_out
        
                                                        
        kernel_size = original_conv.kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            kernel_size = (kernel_size[0], kernel_size[1])
            
        stride = original_conv.stride
        if isinstance(stride, int):
            stride = (stride, stride)
        else:
            stride = (stride[0], stride[1])
            
        padding = original_conv.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        elif isinstance(padding, tuple) and len(padding) >= 2:
            padding = (padding[0], padding[1])
        elif isinstance(padding, tuple) and len(padding) == 1:
            padding = (padding[0], padding[0])
        elif isinstance(padding, str):
                                                                     
            pass
        else:
                              
            padding = (1, 1)
        
                                                     
        new_conv = nn.Conv2d(
            original_conv.in_channels,
            target_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=original_conv.bias is not None
        )
        
                                             
        with torch.no_grad():
            if target_channels <= original_channels:
                                        
                new_conv.weight.data = original_conv.weight.data[:target_channels, :, :, :]
                if new_conv.bias is not None and original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data[:target_channels]
            else:
                                               
                repeat_factor = target_channels // original_channels
                remainder = target_channels % original_channels
                
                weight_parts = []
                bias_parts = []
                
                for i in range(repeat_factor):
                    weight_parts.append(original_conv.weight.data)
                    if original_conv.bias is not None:
                        bias_parts.append(original_conv.bias.data)
                
                if remainder > 0:
                    weight_parts.append(original_conv.weight.data[:remainder, :, :, :])
                    if original_conv.bias is not None:
                        bias_parts.append(original_conv.bias.data[:remainder])
                
                new_conv.weight.data = torch.cat(weight_parts, dim=0) / repeat_factor
                if new_conv.bias is not None and len(bias_parts) > 0:
                    new_conv.bias.data = torch.cat(bias_parts, dim=0) / repeat_factor
        
                           
        self.unet.conv_out = new_conv
    
    def _setup_flash_attention(self):
        """Setup flash attention for better performance."""
        try:
                                                            
            self.unet.set_attn_processor(AttnProcessor2_0())
            print("Flash attention enabled")
        except Exception as e:
            print(f"Could not enable flash attention: {e}")
    
    def _freeze_encoder(self):
        """Freeze the encoder part of the U-Net."""
        print("Freezing U-Net encoder...")
                                          
        for param in self.unet.down_blocks.parameters():
            param.requires_grad = False
        for param in self.unet.mid_block.parameters():
            param.requires_grad = False
                                                       
        
    def _freeze_decoder(self):
        """Freeze the decoder part of the U-Net."""
        print("Freezing U-Net decoder...")
                          
        for param in self.unet.up_blocks.parameters():
            param.requires_grad = False
                                                        
    
    def _unfreeze_cross_attention_layers(self):
        """Unfreeze cross-attention layers for text conditioning fine-tuning."""
        print("Ensuring cross-attention layers are trainable...")
        
                                                                
        def unfreeze_cross_attn_in_block(block):
            if hasattr(block, 'attentions') and block.attentions is not None:
                for attention_module in block.attentions:
                    if hasattr(attention_module, 'transformer_blocks'):
                        for transformer_block in attention_module.transformer_blocks:
                                                              
                            if hasattr(transformer_block, 'attn2'):
                                for param in transformer_block.attn2.parameters():
                                    param.requires_grad = True
                                                                    
                            if hasattr(transformer_block, 'norm2'):
                                for param in transformer_block.norm2.parameters():
                                    param.requires_grad = True
        
                                                 
        for down_block in self.unet.down_blocks:
            unfreeze_cross_attn_in_block(down_block)
        
                                               
        unfreeze_cross_attn_in_block(self.unet.mid_block)
        
                                               
        for up_block in self.unet.up_blocks:
            unfreeze_cross_attn_in_block(up_block)
        
                                                    
        cross_attn_params = 0
        for name, param in self.unet.named_parameters():
            if ('attn2' in name or 'norm2' in name) and param.requires_grad:
                cross_attn_params += param.numel()
        
        print(f"✅ Cross-attention layers unfrozen: {cross_attn_params:,} trainable parameters")
    
    def configure_training_mode(self, mode: str = "full"):
        """
        Configure which parts of the U-Net to train.
        
        Args:
            mode: Training mode - "full", "cross_attention_only", "decoder_only", or "custom"
        """
        print(f"Configuring training mode: {mode}")
        
        if mode == "full":
                              
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.text_projection.parameters():
                param.requires_grad = True
            print("✅ Full model training enabled")
            
        elif mode == "cross_attention_only":
                                     
            for param in self.unet.parameters():
                param.requires_grad = False
                                                                         
            self._unfreeze_cross_attention_layers()
            for param in self.text_projection.parameters():
                param.requires_grad = True
                                                                             
            for param in self.unet.conv_in.parameters():
                param.requires_grad = True
            for param in self.unet.conv_out.parameters():
                param.requires_grad = True
            print("✅ Cross-attention only training enabled")
            
        elif mode == "decoder_only":
                                           
            for param in self.unet.down_blocks.parameters():
                param.requires_grad = False
            for param in self.unet.mid_block.parameters():
                param.requires_grad = True
            for param in self.unet.up_blocks.parameters():
                param.requires_grad = True
            for param in self.unet.conv_out.parameters():
                param.requires_grad = True
            self._unfreeze_cross_attention_layers()
            print("✅ Decoder-only training enabled")
            
                                                        
        for param in self.text_projection.parameters():
            param.requires_grad = True
        for param in self.text_layer_norm.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the pre-trained U-Net with stability improvements.
        
        Args:
            noisy_latent: [batch_size, 8, 27, 27] noisy latent representation
            timesteps: [batch_size] timestep indices
            text_emb: [batch_size, seq_len, 256] text embeddings from TextEncoder
            return_dict: Whether to return a dict (for compatibility)
            
        Returns:
            predicted_noise: [batch_size, 8, 27, 27] predicted noise
        """
                                                           
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        noisy_latent = noisy_latent.to(device=device, dtype=dtype)
        timesteps = timesteps.to(device=device)
        text_emb = text_emb.to(device=device, dtype=dtype)
        
                                                           
        if torch.isnan(noisy_latent).any() or torch.isinf(noisy_latent).any():
            print(f"⚠️ Warning: NaN/Inf detected in noisy_latent input")
            noisy_latent = torch.nan_to_num(noisy_latent, nan=0.0, posinf=1.0, neginf=-1.0)
                                            
            noisy_latent = torch.clamp(noisy_latent, min=-10.0, max=10.0)
        
        if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
            print(f"⚠️ Warning: NaN/Inf detected in text_emb input")
            text_emb = torch.nan_to_num(text_emb, nan=0.0, posinf=1.0, neginf=-1.0)
                                            
            text_emb = torch.clamp(text_emb, min=-10.0, max=10.0)
        
                               
        batch_size = noisy_latent.shape[0]
        assert noisy_latent.shape == (batch_size, self.latent_dim, 27, 27),\
            f"Expected noisy_latent shape [{batch_size}, {self.latent_dim}, 27, 27], got {noisy_latent.shape}"
        assert timesteps.shape == (batch_size,),\
            f"Expected timesteps shape [{batch_size}], got {timesteps.shape}"
        assert text_emb.shape[0] == batch_size and text_emb.shape[2] == self.text_dim,\
            f"Expected text_emb shape [{batch_size}, seq_len, {self.text_dim}], got {text_emb.shape}"
        
                                                                          
        projected_text_emb = self.text_projection(text_emb)
        
                                                 
        projected_text_emb = self.text_layer_norm(projected_text_emb)
        
                                                              
        projected_text_emb = torch.clamp(projected_text_emb, min=-10.0, max=10.0)
        
                                            
        if torch.isnan(projected_text_emb).any() or torch.isinf(projected_text_emb).any():
            print(f"⚠️ Warning: NaN/Inf detected in projected text embeddings")
            projected_text_emb = torch.nan_to_num(projected_text_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
                                                        
        try:
            output = self.unet(
                sample=noisy_latent,
                timestep=timesteps,
                encoder_hidden_states=projected_text_emb,
                return_dict=False
            )
            
                                                
            if isinstance(output, tuple):
                predicted_noise = output[0]
            else:
                predicted_noise = output
            
                                                    
            if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
                print(f"⚠️ Warning: NaN/Inf detected in U-Net output")
                predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0, posinf=1.0, neginf=-1.0)
            
                                              
            predicted_noise = torch.clamp(predicted_noise, min=-50.0, max=50.0)
            
                                            
            assert predicted_noise.shape == noisy_latent.shape,\
                f"Output shape {predicted_noise.shape} doesn't match input shape {noisy_latent.shape}"
            
            return predicted_noise
            
        except Exception as e:
            print(f"Error in U-Net forward pass: {e}")
            print(f"Input shapes: noisy_latent={noisy_latent.shape}, timesteps={timesteps.shape}, text_emb={text_emb.shape}")
            raise
    
    def get_parameter_count(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_training_summary(self):
        """Print a summary of what parts of the model are trainable."""
        total_params = self.get_parameter_count()
        trainable_params = self.get_trainable_parameter_count()
        
        print("\n" + "="*50)
        print("U-NET TRAINING CONFIGURATION SUMMARY")
        print("="*50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params:.1%}")
        
                                   
        cross_attn_params = sum(p.numel() for name, p in self.unet.named_parameters() 
                               if ('attn2' in name or 'norm2' in name) and p.requires_grad)
        text_proj_params = sum(p.numel() for p in self.text_projection.parameters() if p.requires_grad)
        conv_params = sum(p.numel() for p in self.unet.conv_in.parameters() if p.requires_grad) +\
                     sum(p.numel() for p in self.unet.conv_out.parameters() if p.requires_grad)
        
        print(f"\nComponent breakdown:")
        print(f"  Cross-attention: {cross_attn_params:,} trainable")
        print(f"  Text projection: {text_proj_params:,} trainable") 
        print(f"  Input/output convs: {conv_params:,} trainable")
        print("="*50 + "\n")


def create_pokemon_unet(
    config: dict,
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
    freeze_encoder: bool = False,
    freeze_decoder: bool = False
) -> DiffusersUNet:
    """
    Create a pre-trained Pokemon sprite generation U-Net.
    
    Args:
        config: Configuration dictionary with model parameters
        pretrained_model_name: Name of the pre-trained Stable Diffusion model
        freeze_encoder: Whether to freeze the encoder during training
        freeze_decoder: Whether to freeze the decoder during training
        
    Returns:
        Initialized DiffusersUNet model
    """
    model = DiffusersUNet(
        latent_dim=config.get('latent_dim', 8),
        text_dim=config.get('text_dim', 256),
        pretrained_model_name=pretrained_model_name,
        cross_attention_dim=config.get('cross_attention_dim', 768),
        attention_head_dim=config.get('attention_head_dim', 8),
        use_flash_attention=config.get('use_flash_attention', True),
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder
    )
    
    return model
