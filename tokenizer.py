from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

def get_tokenizer(save_path="tokenizer.json"):
    # Create an empty BPE model
    tokenizer = Tokenizer(models.BPE())

    # Add a special blank / padding token
    # "▁" is the blank/pad token
    tokenizer.add_special_tokens(["▁"])   

    # Add character-level tokens: A–Z and space
    tokenizer.add_tokens(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "))

    # Byte-level pre-tokenizer and decoder (handles spacing/bytes cleanly)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    # Optionally remember the id of the blank token
    # We can attach it to the tokenizer object as a custom attribute or just rely on looking it up
    tokenizer.blank_token = "▁"
    tokenizer.blank_token_id = tokenizer.token_to_id("▁")

    # Save to disk so you can reload later
    # Note: verify that save_path ends with .json or is just a filename
    tokenizer.save(save_path)

    return tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    print("Tokenizer created.")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Blank token id: {tokenizer.token_to_id('▁')}")
    
    encoded = tokenizer.encode("HELLO WORLD")
    print(f"Encoded 'HELLO WORLD': {encoded.ids}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")