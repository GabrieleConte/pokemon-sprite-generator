import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):
    """Attention mechanism linking text encoder and image decoder."""
    
    def __init__(self, encoder_dim=256, decoder_dim=256, hidden_dim=256):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.hidden_dim = hidden_dim
        
        # Linear transformations for attention computation
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.decoder_proj = nn.Linear(decoder_dim, hidden_dim)
        self.attention_proj = nn.Linear(hidden_dim, 1)
        
        # Context vector projection
        self.context_proj = nn.Linear(encoder_dim, decoder_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, encoder_outputs, decoder_state, attention_mask=None):
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_outputs: Encoder hidden states [batch_size, seq_len, encoder_dim]
            decoder_state: Current decoder state [batch_size, decoder_dim]
            attention_mask: Mask for padded positions [batch_size, seq_len]
            
        Returns:
            context_vector: Weighted average of encoder outputs [batch_size, decoder_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Project encoder outputs and decoder state to hidden dimension
        encoder_projected = self.encoder_proj(encoder_outputs)  # [batch_size, seq_len, hidden_dim]
        decoder_projected = self.decoder_proj(decoder_state)    # [batch_size, hidden_dim]
        
        # Expand decoder state to match sequence length
        decoder_expanded = decoder_projected.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Compute attention scores
        combined = torch.tanh(encoder_projected + decoder_expanded)  # [batch_size, seq_len, hidden_dim]
        attention_scores = self.attention_proj(combined).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # Compute context vector as weighted average
        context_vector = torch.sum(
            encoder_outputs * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, encoder_dim]
        
        # Project context vector to decoder dimension
        context_vector = self.context_proj(context_vector)  # [batch_size, decoder_dim]
        
        return context_vector, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for more complex attention patterns."""
    
    def __init__(self, encoder_dim=256, decoder_dim=256, hidden_dim=256, num_heads=8):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear transformations for multi-head attention
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.decoder_proj = nn.Linear(decoder_dim, hidden_dim)
        self.value_proj = nn.Linear(encoder_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, decoder_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, encoder_outputs, decoder_state, attention_mask=None):
        """
        Compute multi-head attention.
        
        Args:
            encoder_outputs: Encoder hidden states [batch_size, seq_len, encoder_dim]
            decoder_state: Current decoder state [batch_size, decoder_dim]
            attention_mask: Mask for padded positions [batch_size, seq_len]
            
        Returns:
            context_vector: Attended context vector [batch_size, decoder_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len]
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Project to queries, keys, and values
        queries = self.decoder_proj(decoder_state)  # [batch_size, hidden_dim]
        keys = self.encoder_proj(encoder_outputs)    # [batch_size, seq_len, hidden_dim]
        values = self.value_proj(encoder_outputs)    # [batch_size, seq_len, hidden_dim]
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, 1, seq_len]
        attention_scores = attention_scores.squeeze(2)  # [batch_size, num_heads, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1)  # [batch_size, num_heads, seq_len]
            attention_scores = attention_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights.unsqueeze(2), values)  # [batch_size, num_heads, 1, head_dim]
        attended_values = attended_values.squeeze(2)  # [batch_size, num_heads, head_dim]
        
        # Concatenate heads
        context_vector = attended_values.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)  # [batch_size, hidden_dim]
        
        # Final projection
        context_vector = self.out_proj(context_vector)  # [batch_size, decoder_dim]
        
        return context_vector, attention_weights


class CrossAttention(nn.Module):
    """Cross-attention mechanism for text-to-image generation."""
    
    def __init__(self, text_dim=256, image_dim=256, hidden_dim=256):
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism
        self.attention = AttentionMechanism(
            encoder_dim=text_dim,
            decoder_dim=image_dim,
            hidden_dim=hidden_dim
        )
        
        # Gate mechanism to control attention influence
        self.gate = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_features, image_features, attention_mask=None):
        """
        Apply cross-attention between text and image features.
        
        Args:
            text_features: Text encoder outputs [batch_size, seq_len, text_dim]
            image_features: Image decoder features [batch_size, image_dim]
            attention_mask: Mask for padded text positions [batch_size, seq_len]
            
        Returns:
            enhanced_features: Enhanced image features [batch_size, image_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        # Compute attention context
        context_vector, attention_weights = self.attention(
            text_features, image_features, attention_mask
        )
        
        # Compute gate value
        gate_input = torch.cat([image_features, context_vector], dim=1)
        gate_value = self.gate(gate_input)
        
        # Apply gated attention
        enhanced_features = image_features + gate_value * context_vector
        
        return enhanced_features, attention_weights