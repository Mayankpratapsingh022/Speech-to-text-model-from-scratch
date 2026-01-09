from transformers import GPT2Tokenizer

def get_tokenizer(save_path="tokenizer"):
    """
    Loads the pre-trained GPT-2 tokenizer and adapts it for Speech-to-Text tasks.
    
    This function:
    1. Loads 'gpt2' (Byte-Level BPE).
    2. Adds special tokens required for STT/CTC:
       - <pad>: Padding token (missing in default GPT-2)
       - <ctc_blank>: Explicit blank token for CTC decoding
       - <s>, </s>: Start/End of sentence markers
    3. Saves the tokenizer config to the specified path.
    
    Returns:
        tokenizer: Configured GPT2Tokenizer.
    """
    
    # Load pre-trained GPT-2 tokenizer
    # It already handles Byte-Pair Encoding
    print("Loading pre-trained GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Define special tokens to add
    special_tokens_dict = {
        "pad_token": "<pad>",
        "additional_special_tokens": ["<ctc_blank>", "<s>", "</s>"]
    }
    
    # Add special tokens
    # This resizes the vocabulary
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")
    
    # Explicitly set the pad token if not set (add_special_tokens handles setting the attribute, but good to double check)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
        
    # Save the tokenizer locally so it can be reloaded with the new tokens
    if save_path:
        print(f"Saving tokenizer to {save_path}...")
        tokenizer.save_pretrained(save_path)
        
    return tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    
    print("\nTokenizer Info:")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"Additional tokens: {tokenizer.additional_special_tokens}")
    
    # Test encoding
    text = "Hello world"
    encoded = tokenizer(text)
    print(f"\nTest Encoding '{text}': {encoded['input_ids']}")
    
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: '{decoded}'")