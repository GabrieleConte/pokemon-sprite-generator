from torch import Tensor
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class TextEncoder(nn.Module):
    """
    Encodes text descriptions into a sequence of hidden states.
    This encoder uses a pre-trained BERT model with selective fine-tuning.
    Supports memory-efficient training by only fine-tuning important layers.
    """
    def __init__(self, model_name='google-bert/bert-base-uncased', hidden_dim=768, 
                 finetune_strategy='minimal'):
        """
        Initializes the TextEncoder.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
            hidden_dim (int): The dimensionality of the hidden states.
            finetune_strategy (str): Strategy for fine-tuning BERT layers:
                - 'none': Freeze all BERT parameters (fastest, least memory)
                - 'minimal': Only fine-tune last 2 layers + projection (recommended)
                - 'partial': Fine-tune last 4 layers + projection (balanced)
                - 'full': Fine-tune all layers (most memory intensive)
        """
        super(TextEncoder, self).__init__()
        
        self.finetune_strategy = finetune_strategy
        
        print(f"Loading text encoder: {model_name}")
        print(f"Fine-tuning strategy: {finetune_strategy}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Apply selective fine-tuning strategy
        self._apply_finetune_strategy()

        self.bert_hidden_size = self.bert.config.hidden_size
        print(f"BERT hidden size: {self.bert_hidden_size}")
        
        # A linear layer to project BERT's output to our desired hidden dimension
        if self.bert_hidden_size != hidden_dim:
            self.projection = nn.Linear(self.bert_hidden_size, hidden_dim)
            print(f"Added projection layer: {self.bert_hidden_size} -> {hidden_dim}")
        else:
            self.projection = nn.Identity()
            print(f"No projection needed: {self.bert_hidden_size} dimensions")
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Always keep projection and layer norm trainable
        for param in self.projection.parameters():
            param.requires_grad = True
        for param in self.layer_norm.parameters():
            param.requires_grad = True
        
        # Print parameter summary
        self._print_parameter_summary()
    
    def _apply_finetune_strategy(self):
        """Apply the selected fine-tuning strategy to BERT parameters."""
        total_layers = len(self.bert.encoder.layer)
        
        if self.finetune_strategy == 'none':
            # Freeze all BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False
            print("✅ Frozen all BERT parameters")
            
        elif self.finetune_strategy == 'minimal':
            # Freeze all except last 2 layers + pooler
            for param in self.bert.parameters():
                param.requires_grad = False
            
            # Unfreeze last 2 layers
            for layer_idx in range(max(0, total_layers - 2), total_layers):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
            
            # Unfreeze pooler for better sentence representations
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
            
            print(f"✅ Fine-tuning last 2 layers (out of {total_layers}) + pooler")
            
        elif self.finetune_strategy == 'partial':
            # Freeze all except last 4 layers + pooler
            for param in self.bert.parameters():
                param.requires_grad = False
            
            # Unfreeze last 4 layers
            for layer_idx in range(max(0, total_layers - 4), total_layers):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
            
            # Unfreeze pooler
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
            
            print(f"✅ Fine-tuning last 4 layers (out of {total_layers}) + pooler")
            
        elif self.finetune_strategy == 'full':
            # Fine-tune all layers
            for param in self.bert.parameters():
                param.requires_grad = True
            print(f"✅ Fine-tuning all {total_layers} layers")
            
        else:
            raise ValueError(f"Unknown finetune_strategy: {self.finetune_strategy}")
    
    def _print_parameter_summary(self):
        """Print a summary of trainable vs frozen parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        bert_trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        projection_trainable = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
        layernorm_trainable = sum(p.numel() for p in self.layer_norm.parameters() if p.requires_grad)
        
        print(f"📊 Parameter Summary:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"   Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"   BERT trainable: {bert_trainable:,}")
        print(f"   Projection trainable: {projection_trainable:,}")
        print(f"   LayerNorm trainable: {layernorm_trainable:,}")
        
        # Memory estimate (rough)
        memory_mb = (trainable_params * 4 * 3) / (1024 * 1024)  # 4 bytes per param, 3x for gradients/optimizer
        print(f"   Estimated training memory: ~{memory_mb:.1f} MB")

    def forward(self, text_list):
        """
        Forward pass for the TextEncoder.

        Args:
            text_list (list of str): A list of text descriptions.

        Returns:
            torch.Tensor: The encoded text as a sequence of hidden states.
        """
        # Tokenize the input text with longer max length for more detailed descriptions
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        # Move inputs to the correct device
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        
        # Get hidden states from BERT
        outputs : BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**inputs)
        bert_hidden_states = outputs.last_hidden_state
        
        # Project the hidden states to the desired dimension
        projected_states = self.projection(bert_hidden_states)
        
        # Apply layer normalization for stability
        normalized_states = self.layer_norm(projected_states)
        
        return normalized_states
