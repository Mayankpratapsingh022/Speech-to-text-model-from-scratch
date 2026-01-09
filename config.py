import os

# --- Paths ---
# Assuming the script is run from the project root
PROJECT_ROOT = os.getcwd()
DATASET_DIR = os.path.join(PROJECT_ROOT, "LJSpeech-1.1")
WAVS_DIR = os.path.join(DATASET_DIR, "wavs")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

# Tokenizer path (saved directory)
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer")

# --- Audio Config ---
SAMPLE_RATE = 16000  # Target sample rate (LJSpeech is 22050Hz, usually resampled to 16k for STT)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80  # Common for MelSpectrograms

# --- Training Config ---
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
MAX_AUDIO_LENGTH = 16000 * 10  # e.g., limit to 10 seconds to avoid OOM (optional)
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.1

# --- Device Config ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Text Config ---
# Special tokens already handled by tokenizer files, but good to have reference if needed
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CTC_BLANK_TOKEN = "<ctc_blank>"

# --- Model Config ---
INPUT_CHANNELS = 1
HIDDEN_DIM = 16
EMBEDDING_DIM = 32
STRIDES = (6, 8, 4, 2)
KERNEL_SIZE = 8
INITIAL_POOLING_KERNEL = 2

# --- Transformer Config ---
NUM_HEADS = 4
NUM_LAYERS = 6
MAX_SEQ_LENGTH = 1000  # Max length of downsampled sequence (audio)
FF_HIDDEN_MULT = 4
DROPOUT = 0.1

# --- VQ / RVQ Config ---
CODEBOOK_SIZE = 1024
NUM_CODEBOOKS = 4
COMMITMENT_COST = 0.25

# --- Training / Loss Config ---
VQ_INITIAL_LOSS_WEIGHT = 10.0
VQ_WARMUP_STEPS = 1000
VQ_FINAL_LOSS_WEIGHT = 0.5
NUM_EPOCHS = 1000
NUM_BATCH_REPEATS = 1
LEARNING_RATE = 0.005
MODEL_ID = "test37"
LOG_DIR = f"runs/{MODEL_ID}"
MODELS_DIR = f"models/{MODEL_ID}"

import os
os.makedirs(MODELS_DIR, exist_ok=True)
