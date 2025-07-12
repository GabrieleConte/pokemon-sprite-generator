from torch import Tensor
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class TextEncoder(nn.Module):
    """
    Encodes text descriptions into a sequence of hidden states.
    This encoder uses a pre-trained BERT model and a Transformer Encoder.
    """
    def __init__(self, model_name='prajjwal1/bert-mini', hidden_dim=256, nhead=4, num_encoder_layers=2):
        """
        Initializes the TextEncoder.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
            hidden_dim (int): The dimensionality of the hidden states.
            nhead (int): The number of attention heads in the Transformer encoder.
            num_encoder_layers (int): The number of layers in the Transformer encoder.
        """
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # We want to fine-tune the BERT model
        for param in self.bert.parameters():
            param.requires_grad = True

        self.bert_hidden_size = self.bert.config.hidden_size
        
        # A linear layer to project BERT's output to our desired hidden dimension
        self.projection = nn.Linear(self.bert_hidden_size, hidden_dim)
        
        # Custom Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, text_list):
        """
        Forward pass for the TextEncoder.

        Args:
            text_list (list of str): A list of text descriptions.

        Returns:
            torch.Tensor: The encoded text as a sequence of hidden states.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Move inputs to the correct device
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        
        # Get hidden states from BERT
        outputs : BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**inputs)
        bert_hidden_states = outputs.last_hidden_state
        
        # Project the hidden states to the desired dimension
        projected_states = self.projection(bert_hidden_states)
        
        # Pass through the Transformer Encoder
        # Transformer expects (seq_len, batch_size, hidden_dim)
        projected_states : Tensor = projected_states.permute(1, 0, 2)
        encoder_output = self.transformer_encoder(projected_states)
        
        # Permute back to (batch_size, seq_len, hidden_dim)
        encoder_output : Tensor = encoder_output.permute(1, 0, 2)
        
        return encoder_output
