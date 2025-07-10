import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import re


class SimpleTokenizer:
    """Simple tokenizer that doesn't require internet access."""
    
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        
        # Build a simple vocabulary
        self.special_tokens = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.cls_token: 2,
            self.sep_token: 3
        }
        
        # Create a simple word-to-id mapping
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        
        # Add common words (this is a simplified vocabulary)
        common_words = [
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'pokemon', 'pikachu', 'charizard', 'blastoise', 'venusaur', 'fire', 'water', 'grass', 'electric',
            'dragon', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel', 'psychic', 'ice', 'dark', 'fairy',
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray',
            'small', 'large', 'big', 'tiny', 'huge', 'tall', 'short', 'wide', 'thin', 'thick',
            'mouse', 'cat', 'dog', 'bird', 'fish', 'turtle', 'lizard', 'snake', 'frog', 'bear',
            'wings', 'tail', 'ears', 'eyes', 'nose', 'mouth', 'teeth', 'claws', 'paws', 'feet',
            'flame', 'fire', 'water', 'ice', 'lightning', 'thunder', 'wind', 'earth', 'stone', 'metal',
            'spikes', 'shell', 'armor', 'fur', 'scales', 'feathers', 'pattern', 'stripes', 'spots',
            'powerful', 'strong', 'fast', 'slow', 'heavy', 'light', 'fierce', 'gentle', 'friendly', 'angry'
        ]
        
        current_id = len(self.special_tokens)
        for word in common_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        # Fill remaining vocabulary with dummy tokens
        while current_id < vocab_size:
            dummy_token = f'[UNUSED{current_id}]'
            self.word_to_id[dummy_token] = current_id
            self.id_to_word[current_id] = dummy_token
            current_id += 1
    
    def tokenize(self, text):
        """Tokenize text into words."""
        # Simple tokenization - split by spaces and punctuation
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs."""
        return [self.word_to_id.get(token, self.word_to_id[self.unk_token]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """Convert IDs to tokens."""
        return [self.id_to_word.get(id, self.unk_token) for id in ids]
    
    def __call__(self, texts, max_length=128, padding='max_length', truncation=True, return_tensors='pt'):
        """Tokenize texts and return tensors."""
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # Tokenize
            tokens = self.tokenize(text)
            
            # Add special tokens
            tokens = [self.cls_token] + tokens + [self.sep_token]
            
            # Convert to IDs
            ids = self.convert_tokens_to_ids(tokens)
            
            # Truncate if necessary
            if truncation and len(ids) > max_length:
                ids = ids[:max_length-1] + [self.word_to_id[self.sep_token]]
            
            # Create attention mask
            attention_mask = [1] * len(ids)
            
            # Pad if necessary
            if padding == 'max_length':
                pad_length = max_length - len(ids)
                ids.extend([self.word_to_id[self.pad_token]] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            input_ids.append(ids)
            attention_masks.append(attention_mask)
        
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_masks
            }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Feed-forward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src


class TextEncoder(nn.Module):
    """Text encoder using BERT-mini embeddings and transformer layers."""
    
    def __init__(self, vocab_size=30522, embedding_dim=256, num_heads=8, 
                 num_layers=6, hidden_dim=512, dropout=0.1, max_seq_length=128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Initialize with a simple tokenizer (no internet required)
        self.tokenizer = SimpleTokenizer(vocab_size)
        
        # Embedding layer (will be initialized with BERT-mini weights)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with BERT-mini weights
        self._init_bert_embeddings()
    
    def _init_bert_embeddings(self):
        """Initialize embedding layer with pre-trained weights."""
        # Since we don't have internet access, initialize randomly
        print("Initializing embeddings randomly (BERT-mini weights would be used with internet access)")
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of text encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            encoder_outputs: Contextual hidden states [batch_size, seq_len, embedding_dim]
            pooled_output: Pooled representation [batch_size, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.transpose(0, 1)  # [seq_len, batch_size, embedding_dim]
        
        # Positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Create padding mask for transformer
        if attention_mask is not None:
            # Convert attention mask to key padding mask (True for padded positions)
            key_padding_mask = ~attention_mask.bool()  # [batch_size, seq_len]
        else:
            key_padding_mask = None
        
        # Pass through transformer layers
        output = embedded
        for layer in self.transformer_layers:
            output = layer(output, src_key_padding_mask=key_padding_mask)
        
        # Transpose back to [batch_size, seq_len, embedding_dim]
        output = output.transpose(0, 1)
        
        # Pooled output (mean pooling over sequence length, excluding padding)
        if attention_mask is not None:
            # Mask out padded positions
            mask_expanded = attention_mask.unsqueeze(-1).expand(output.size()).float()
            sum_embeddings = torch.sum(output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(output, dim=1)
        
        return output, pooled_output
    
    def tokenize(self, texts):
        """Tokenize text inputs."""
        if isinstance(texts, str):
            texts = [texts]
        
        encoding = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding['input_ids'], encoding['attention_mask']