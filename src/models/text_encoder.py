from torch import Tensor
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class TextEncoder(nn.Module):
    """
    Encodes text descriptions into a sequence of hidden states.
    This encoder uses a pre-trained BERT model with direct projection.
    Now supports BERT-base for richer text representations.
    """
    def __init__(self, model_name='google-bert/bert-base-uncased', hidden_dim=768):
        """
        Initializes the TextEncoder.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
            hidden_dim (int): The dimensionality of the hidden states.
        """
        super(TextEncoder, self).__init__()
        
        print(f"Loading text encoder: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # We want to fine-tune the BERT model for better domain adaptation
        for param in self.bert.parameters():
            param.requires_grad = True

        self.bert_hidden_size = self.bert.config.hidden_size
        print(f"BERT hidden size: {self.bert_hidden_size}")
        
        # A linear layer to project BERT's output to our desired hidden dimension
        # For BERT-large, this might be identity if hidden_dim matches bert_hidden_size
        if self.bert_hidden_size != hidden_dim:
            self.projection = nn.Linear(self.bert_hidden_size, hidden_dim)
            print(f"Added projection layer: {self.bert_hidden_size} -> {hidden_dim}")
        else:
            self.projection = nn.Identity()
            print(f"No projection needed: {self.bert_hidden_size} dimensions")
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        print(f"Text encoder initialized with {sum(p.numel() for p in self.parameters()):,} parameters")

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
