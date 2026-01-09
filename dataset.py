import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import config
from tokenizer import get_tokenizer

class LJSpeechDataset(Dataset):
    """
    Dataset class for LJSpeech.
    Reads the metadata.csv and loads audio on the fly.
    """
    def __init__(self, metadata_path, wavs_dir, tokenizer, sample_rate=16000, transform=None):
        self.metadata = pd.read_csv(
            metadata_path, 
            sep='|', 
            header=None, 
            names=['ID', 'Transcription', 'NormalizedTranscription'],
            quoting=3  # Quote minimal to avoid errors with quotes in text
        )
        self.metadata.dropna(inplace=True) # basic cleanup
        
        self.wavs_dir = wavs_dir
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_id = row['ID']
        # Use normalized transcription if available, else original
        text = row['NormalizedTranscription'] if pd.notna(row['NormalizedTranscription']) else row['Transcription']
        
        wav_path = os.path.join(self.wavs_dir, f"{file_id}.wav")
        
        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Mono check (LJSpeech is mono, but good ensuring shape is [1, T] or [T])
        # We usually want [T] or [1, T]. Let's keep it [1, T] or squeeze if expected.
        # STT models often expect [C, T] or just [T]. Let's return [Channel, Time] for flexibility.
        
        # Apply transforms (e.g. MelSpectrogram) if any
        # For now, we return raw waveform as requested by user's snippet preference, 
        # but configured for correct sample rate.
        
        # Text Tokenization
        # self.tokenizer is expected to be the GPT2Tokenizer compatible instance
        # we treat text as target.
        encoded = self.tokenizer(text, return_attention_mask=False) 
        # encoded is a dict or BatchEncoding. We need 'input_ids'.
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        
        return {
            "file_id": file_id,
            "audio": waveform.squeeze(0), # Return (T,) for easier padding in collate
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
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        
        return {
            "audio": padded_audio,
            "input_ids": padded_input_ids,
            "text": texts,
            "audio_lengths": torch.tensor([len(x) for x in audio_list], dtype=torch.long),
            "target_lengths": torch.tensor([len(x) for x in input_ids_list], dtype=torch.long)
        }

def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    # 1. Load Tokenizer
    tokenizer = get_tokenizer(save_path=config.TOKENIZER_PATH)
    
    # 2. Initialize Dataset
    full_dataset = LJSpeechDataset(
        metadata_path=config.METADATA_PATH,
        wavs_dir=config.WAVS_DIR,
        tokenizer=tokenizer,
        sample_rate=config.SAMPLE_RATE
    )
    
    # 3. Split Dataset
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * config.TRAIN_SPLIT)
    val_size = dataset_size - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # 4. Create Dataloaders
    collator = VoiceCollator(tokenizer)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collator,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, # Can be lower for val
        collate_fn=collator,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader


