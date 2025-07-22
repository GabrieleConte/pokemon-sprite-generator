from torch import Tensor
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class TextEncoder(nn.Module):
    """
    Encodes text descriptions into a sequence of hidden states.
    This encoder uses a pre-trained BERT model with direct projection.
    """
    def __init__(self, model_name='prajjwal1/bert-mini', hidden_dim=256):
        """
        Initializes the TextEncoder.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
            hidden_dim (int): The dimensionality of the hidden states.
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
        
        return projected_states
