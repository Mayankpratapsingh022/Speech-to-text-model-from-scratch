import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tokenizer import get_tokenizer
import config
from datasets import load_dataset
from datasets import Audio

class HuggingFaceSpeechDataset(Dataset):
    """
    Dataset class for Hugging Face speech datasets.
    Loads audio and text from Hugging Face datasets.
    """
    def __init__(self, dataset_name, tokenizer, sample_rate=16000, split="train", transform=None):
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name)
        
        # Handle DatasetDict (when no split is specified) vs Dataset
        if hasattr(dataset, 'keys'):  # It's a DatasetDict
            if split in dataset:
                self.dataset = dataset[split]
            else:
                # If split doesn't exist, use the first available split
                available_splits = list(dataset.keys())
                if available_splits:
                    self.dataset = dataset[available_splits[0]]
                else:
                    raise ValueError(f"No splits available in dataset {dataset_name}")
        else:
            # It's already a Dataset
            self.dataset = dataset
        
        # Cast audio column to Audio with decoding
        self.dataset = self.dataset.cast_column("audio", Audio(decode=True))
        
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get audio array and sample rate from Hugging Face dataset
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        
        # Convert to torch tensor
        waveform = torch.tensor(audio_array, dtype=torch.float32)
        
        # Ensure mono (squeeze if stereo)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)  # Average channels if stereo
        
        # Resample if necessary
        if sr != self.sample_rate:
            # Add channel dimension for resampler: (T,) -> (1, T)
            waveform = waveform.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            waveform = waveform.squeeze(0)  # Back to (T,)
        
        # Get text - try common column names
        text = item.get("text", item.get("transcription", item.get("sentence", "")))
        if text is None:
            text = ""
        
        # Get file_id if available, otherwise use index
        file_id = item.get("file", item.get("id", f"sample_{idx}"))
        
        # Text Tokenization
        encoded = self.tokenizer.encode(text) 
        input_ids = torch.tensor(encoded.ids, dtype=torch.long)
        
        return {
            "file_id": str(file_id),
            "audio": waveform,  # Return (T,) for easier padding in collate
            "input_ids": input_ids,
            "text": text
        }

def collate_fn(batch):
    """
    Custom collate function to pad audio and text to max length in the batch.
    """
    # Filter out None/failed items if you implement error handling
    batch = [item for item in batch if item is not None]
    
    # 1. Pad Audio
    # Audio is (T,)
    audio_list = [item['audio'] for item in batch]
    input_ids_list = [item['input_ids'] for item in batch]
    texts = [item['text'] for item in batch]
    file_ids = [item['file_id'] for item in batch]
    
    # Pad audio with 0
    padded_audio = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    # Result: (B, T_max)
    
    # Pad text with tokenizer's pad token id
    # We need to access tokenizer from somewhere, or assume a fixed ID. 
    # Usually passed in or we use a convention. 
    # Let's hope the tokenizer was set globally or we use a safe default.
    # Better: pass pad_id or assume config.
    # We will assume the tokenizer has a pad_token_id. 
    # Since we don't have access to the tokenizer instance here easily without passing it,
    # let's assume valid ID is available or use 0 if <pad> is at 0 (which we tried to ensure).
    # Re-instantiating tokenizer is cheap if cached, or we can check the first item if we stored it?
    # Actually, let's grab the pad_id from the tokenizer in the dataset if possible? 
    # No, collate is function.
    # We'll use a hardcoded value or grab from config if we are sure.
    # User's tokenizer snippet printed: Pad token: <pad> (id: 50257) for GPT2!
    # Wait, GPT2 default pad is often missing, we added it.
    # The ID depends on the vocab size.
    # Better approach: We should probably store pad_id in the collate_fn or a partial.
    
    # For now, let's try to get it from the dataset instance attached to the batch? No.
    # Let's import the tokenizer and check? Or pass it to collate_fn wrapper.
    # Simplest for now: Use a placeholder or import config if we stored it there?
    # We didn't store the exact ID in config.
    # Let's modify get_dataloaders to use a partial or a class-based collator.
    
    # For simplicity in this script, I will just use 0 or -100 or raise an issue.
    # Actually, let's use a class `VoiceCollator` which takes tokenizer.
    pass # Will be implemented in the class below.

class VoiceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        audio_list = [item['audio'] for item in batch]
        input_ids_list = [item['input_ids'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Pad audio (B, T)
        padded_audio = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        
        # Pad IDs
        pad_id = self.tokenizer.token_to_id(config.PAD_TOKEN)
        # If not found, default to 0 (but it should be found)
        if pad_id is None:
            pad_id = 0
            
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        
        return {
            "audio": padded_audio,
            "input_ids": padded_input_ids,
            "text": texts,
            "audio_lengths": torch.tensor([len(x) for x in audio_list], dtype=torch.long),
            "target_lengths": torch.tensor([len(x) for x in input_ids_list], dtype=torch.long)
        }

def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, dataset_name=None):
    if dataset_name is None:
        dataset_name = getattr(config, 'HF_DATASET_NAME', "m-aliabbas/idrak_timit_subsample1")
    # 1. Load Tokenizer
    tokenizer = get_tokenizer(save_path=config.TOKENIZER_PATH)
    
    # 2. Initialize Datasets - try to get train/val splits if available, otherwise split manually
    try:
        # Try to load train and validation splits directly
        train_dataset = HuggingFaceSpeechDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            sample_rate=config.SAMPLE_RATE,
            split="train"
        )
        val_dataset = HuggingFaceSpeechDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            sample_rate=config.SAMPLE_RATE,
            split="validation"
        )
    except (ValueError, KeyError, Exception):
        # If splits don't exist or there's an error, load full dataset and split manually
        # Try to load without specifying split (will use first available split)
        try:
            full_dataset = HuggingFaceSpeechDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                sample_rate=config.SAMPLE_RATE,
                split="train"  # Will fallback to first available split if "train" doesn't exist
            )
        except Exception:
            # If that fails, try loading the dataset directly and let it handle the split
            from datasets import load_dataset as hf_load_dataset
            ds = hf_load_dataset(dataset_name)
            # Get the first available split
            if hasattr(ds, 'keys'):
                first_split = list(ds.keys())[0]
                full_dataset = HuggingFaceSpeechDataset(
                    dataset_name=dataset_name,
                    tokenizer=tokenizer,
                    sample_rate=config.SAMPLE_RATE,
                    split=first_split
                )
            else:
                raise ValueError(f"Could not load dataset {dataset_name} with any split")
        
        # Split Dataset
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * config.TRAIN_SPLIT)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 3. Create Dataloaders
    collator = VoiceCollator(tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collator,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, # Can be lower for val
        collate_fn=collator,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader


