import torch
from typing import List
from transformers import BertTokenizer

def get_tokenized_text(text_list: List[str], model_name: str = 'prajjwal1/bert-mini') -> List[List[str]]:
    """
    Get the actual tokens from text using the same tokenizer as the model.
    
    Args:
        text_list: List of text strings
        model_name: Name of the BERT model to use for tokenization
    
    Returns:
        List of token lists, one for each input text
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    tokenized_texts = []
    for text in text_list:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        
        # Add special tokens to match model behavior
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        tokenized_texts.append(tokens)
    
    return tokenized_texts

def get_attention_tokens_mapping(text_list: List[str], 
                               attention_weights: List[torch.Tensor],
                               model_name: str = 'prajjwal1/bert-mini') -> List[List[str]]:
    """
    Get proper token mapping for attention analysis.
    
    Args:
        text_list: Original text inputs
        attention_weights: Attention weights from the model
        model_name: BERT model name used
    
    Returns:
        List of token lists that match the attention sequence length
    """
    # Get the sequence length from attention weights
    seq_len = attention_weights[0].shape[2] if attention_weights else 128
    
    # Get actual tokens
    tokenized_texts = get_tokenized_text(text_list, model_name)
    
    # Pad or truncate to match attention sequence length
    result = []
    for tokens in tokenized_texts:
        if len(tokens) < seq_len:
            # Pad with [PAD] tokens
            padded_tokens = tokens + ['[PAD]'] * (seq_len - len(tokens))
        else:
            # Truncate to fit
            padded_tokens = tokens[:seq_len]
        result.append(padded_tokens)
    
    return result

def demo_proper_tokenization():
    """
    Demonstrate how to use proper tokenization with attention analysis.
    """
    from src.models.pokemon_generator import PokemonSpriteGenerator
    from src.utils.attention_utils import analyze_attention_focus
    
    # Initialize model
    generator = PokemonSpriteGenerator()
    
    # Test texts
    text_descriptions = [
        "A small electric mouse pokemon with yellow fur and red cheeks"
    ]
    
    # Generate with attention
    with torch.no_grad():
        generated_images = generator(text_descriptions)
        attention_weights = generator.get_attention_weights()
    
    # Get proper tokens
    proper_tokens = get_attention_tokens_mapping(text_descriptions, attention_weights)
    
    print("Original text:", text_descriptions[0])
    print("Tokenized:", proper_tokens[0])
    print("Attention sequence length:", attention_weights[0].shape[2])
    print("Token count:", len(proper_tokens[0]))
    
    # Analyze with proper tokens
    analysis = analyze_attention_focus(
        attention_weights,
        proper_tokens[0],  # Use first text's tokens
        threshold=0.05
    )
    
    print("\nAttention analysis with proper tokens:")
    for layer_name, layer_analysis in analysis.items():
        print(f"{layer_name}: {layer_analysis['most_attended_token']}")

if __name__ == "__main__":
    demo_proper_tokenization()
