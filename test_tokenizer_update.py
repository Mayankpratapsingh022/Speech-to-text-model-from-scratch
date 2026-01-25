import torch
from tokenizer import get_tokenizer
from dataset import VoiceCollator
import config

def test_tokenizer_basic():
    print("\n--- Testing Tokenizer Basic ---")
    tokenizer = get_tokenizer()
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    
    text = "HELLO WORLD"
    encoded = tokenizer.encode(text)
    print(f"Input: '{text}'")
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Decoded: '{tokenizer.decode(encoded.ids)}'")
    
    # Check special tokens
    pad_id = tokenizer.token_to_id("▁")
    print(f"Pad Token ID: {pad_id}")
    assert pad_id is not None, "Pad token ▁ not found!"
    
    # Check spacing
    text2 = "A B C"
    enc2 = tokenizer.encode(text2)
    print(f"'{text2}' -> {enc2.ids}")
    assert len(enc2.ids) >= 3

def test_dataset_collator_mock():
    print("\n--- Testing Dataset Collator Mock ---")
    tokenizer = get_tokenizer()
    collator = VoiceCollator(tokenizer)
    
    # Create fake batch
    # Audio: (T,)
    # Text: str
    # Input Ids: Tensor
    
    batch = [
        {
            "audio": torch.randn(1000), 
            "input_ids": torch.tensor(tokenizer.encode("A").ids), 
            "text": "A"
        },
        {
            "audio": torch.randn(2000), 
            "input_ids": torch.tensor(tokenizer.encode("AB").ids), 
            "text": "AB"
        }
    ]
    
    collated = collator(batch)
    
    print("Collated keys:", collated.keys())
    print("Audio shape:", collated["audio"].shape)
    print("Input IDs shape:", collated["input_ids"].shape)
    print("Input IDs:", collated["input_ids"])
    
    # Check padding
    # Second item "AB" is longer (2 tokens) than "A" (1 token).
    # First item should be padded.
    # Pad ID should be 0 (from ▁)
    pad_id = tokenizer.token_to_id("▁")
    
    # Check first row, 2nd token should be pad_id
    if collated["input_ids"][0, 1] != pad_id:
        print(f"WARNING: Expected pad id {pad_id} at [0,1], got {collated['input_ids'][0, 1]}")
    else:
        print("Padding check passed.")

if __name__ == "__main__":
    test_tokenizer_basic()
    test_dataset_collator_mock()
