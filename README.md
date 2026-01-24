# Speech-to-Text Model from Scratch

This repository contains an end-to-end implementation of a Speech-to-Text (ASR) system built from scratch using PyTorch. The architecture integrates a convolutional downsampling frontend, a Transformer encoder, and a Residual Vector Quantizer (RVQ), optimized for training on the LJSpeech dataset.

## Architecture Overview

The model (`transcribe.py`) consists of three main components:

1.  **Downsampling Network (`downsampling.py`)**:
    A convolutional neural network that processes raw audio waveforms. It uses a stack of residual blocks with strided convolutions to reduce the temporal resolution by a factor of approximately 768x, transforming high-frequency audio samples into dense feature representations.

2.  **Transformer Encoder (`transformer.py`)**:
    A standard Transformer encoder with multi-head self-attention and feed-forward layers. It processes the downsampled features to capture long-range dependencies and contextual information essential for speech recognition.

3.  **Residual Vector Quantizer (`rvq.py` & `vector_quantizer.py`)**:
    A multi-stage quantization module that discretizes the high-dimensional feature vectors. It uses a codebook to map continuous features to discrete tokens, facilitating training objectives similar to VQ-VAE.

## Directory Structure

*   `config.py`: Central configuration file for hyperparameters (sample rate, batch size, model dimensions, etc.).
*   `dataset.py`: PyTorch Dataset implementation for loading and preprocessing LJSpeech audio and transcripts.
*   `tokenizer/`: Directory containing the pre-trained BPE tokenizer with specific special tokens for ASR.
*   `train.py`: Main training loop implementation, including logging and checkpointing.
*   `main.py`: Entry point script to launch the training process.
*   `ctc_utils.py`: Helper functions for computing CTC Loss and performing greedy decoding.
*   `requirements.txt`: List of Python dependencies.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Mayankpratapsingh022/Speech-to-text-model-from-scratch.git
    cd Speech-to-text-model-from-scratch
    ```

2.  Install dependencies:
    ```bash
    apt-get update && apt-get install -y ffmpeg
    ```


    ```bash
    pip install -r requirements.txt
    ```

3.  **Dataset Preparation**:
    Ensure the LJSpeech-1.1 dataset is downloaded and extracted in the project root directory. The expected path structure is:
    ```
    ./LJSpeech-1.1/
        metadata.csv
        wavs/
            LJ001-0001.wav
            ...
    ```

## Training

To start training the model, run the main script:

```bash
python main.py
```

### Configuration

You can modify training hyperparameters in `config.py`. Key parameters include:

*   `BATCH_SIZE`: Number of samples per training step.
*   `LEARNING_RATE`: Initial learning rate for the Adam optimizer.
*   `NUM_EPOCHS`: Total number of training epochs.
*   `DEVICE`: Hardware accelerator settings (automatically detects CUDA or MPS).

### Logging and Monitoring

Training metrics (CTC Loss, VQ Loss) are logged using TensorBoard. To view the training progress:

```bash
tensorboard --logdir runs/
```

Checkpoints are saved automatically to the `models/` directory. The evaluation step runs at the end of each epoch and prints sample predictions alongside ground truth transcripts to the console.

## Usage (Inference)

The training script automatically performs validation at the end of each epoch using the `evaluate` function in `train.py`. This function loads a validation batch, runs the forward pass, and decodes the output probabilities using greedy CTC decoding.

To run inference on a specific audio file (example usage):

```python
import torch
from transcribe import TranscribeModel
from tokenizer import get_tokenizer
from dataset import load_audio  # Assumes helper exists or use torchaudio directly

# Load Model
model = TranscribeModel.load("models/test37/model_latest.pth")
model.eval()

# Load Tokenizer
tokenizer = get_tokenizer()

# Load Audio
waveform, sr = torchaudio.load("path/to/audio.wav")
# ... Resample to 16000Hz and normalize ...

# Inference
with torch.no_grad():
    log_probs, _ = model(waveform.unsqueeze(0))
    decoded_text = decode_ctc_output(log_probs, tokenizer, blank_token_id)
    print(decoded_text)
```
