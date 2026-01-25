import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
import config
from dataset import get_dataloaders
from transcribe import TranscribeModel
from tokenizer import get_tokenizer
from ctc_utils import run_loss_function, decode_ctc_output

def train_model():
    # Setup paths
    log_dir = config.LOG_DIR
    models_dir = config.MODELS_DIR
    
    # reset log dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # Tokenizer & Special Tokens
    tokenizer = get_tokenizer()
    # We need to know which token ID is the blank.
    # Current setup: "▁" is the blank/pad token.
    blank_token = tokenizer.token_to_id(config.CTC_BLANK_TOKEN)
    if blank_token is None:
        print(f"Warning: {config.CTC_BLANK_TOKEN} not found, using 0 as fallback.")
        blank_token = 0

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load or Create Model
    latest_path = f"{models_dir}/model_latest.pth"
    steps = config.starting_steps if hasattr(config, 'starting_steps') else 0
    
    if os.path.exists(latest_path):
        print(f"Loading model from {latest_path}")
        model = TranscribeModel.load(latest_path, map_location=device).to(device)
    else:
        print("Creating new model...")
        model = TranscribeModel(
            num_codebooks=config.NUM_CODEBOOKS,
            codebook_size=config.CODEBOOK_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            vocab_size=tokenizer.get_vocab_size(),
            strides=config.STRIDES,
            initial_mean_pooling_kernel_size=config.INITIAL_POOLING_KERNEL,
            num_transformer_layers=config.NUM_LAYERS,
            max_seq_length=config.MAX_SEQ_LENGTH,
            num_heads=config.NUM_HEADS,
        ).to(device)
        steps = 0

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Get Dataloaders
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE, # Defined in config for ease
        num_workers=config.NUM_WORKERS 
    )
    # The snippet used 'num_examples' and 'num_batch_repeats' logic, which is fine to keep inside logic
    # but dataset.py handles the core loading.
    
    ctc_losses = []
    vq_losses = []

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for idx, batch in enumerate(train_loader):
            for _ in range(config.NUM_BATCH_REPEATS):
                # audio: (B, T_audio)
                audio = batch["audio"].to(device)
                # input_ids: (B, T_text)
                target = batch["input_ids"].to(device)

                # Ensure audio is at least as long as target along time for CTC sanity
                # Note: Model downsamples audio! 
                # Target length is num_tokens. Output length is T / downsample_factor.
                # CTC requires Output_Length >= Target_Length
                # If T_audio is too short relative to text, we might have issues.
                # We can verify roughly or let CTCLoss error out (zero_infinity handles it gracefully-ish)
                
                optimizer.zero_grad()
                output, vq_loss = model(audio)          # (B, T', vocab)

                # Compute loss
                ctc_loss = run_loss_function(output, target, blank_token)

                # VQ warmup
                vq_loss_weight = max(
                    config.VQ_FINAL_LOSS_WEIGHT,
                    config.VQ_INITIAL_LOSS_WEIGHT
                    - (config.VQ_INITIAL_LOSS_WEIGHT - config.VQ_FINAL_LOSS_WEIGHT)
                    * (steps / config.VQ_WARMUP_STEPS),
                )

                if vq_loss is None:
                    loss = ctc_loss
                else:
                    loss = ctc_loss + vq_loss_weight * vq_loss

                if torch.isinf(loss) or torch.isnan(loss):
                    print("Loss is invalid, skipping step", audio.shape, target.shape)
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                ctc_losses.append(ctc_loss.item())
                if vq_loss is not None:
                    vq_losses.append(vq_loss.item())
                
                steps += 1

                if steps % 20 == 0:
                    avg_ctc_loss = sum(ctc_losses) / len(ctc_losses)
                    avg_vq_loss = sum(vq_losses) / len(vq_losses) if vq_losses else 0.0
                    avg_loss = avg_ctc_loss + vq_loss_weight * avg_vq_loss

                    print(
                        f"Epoch {epoch+1}, Step {steps}, Batch {idx+1}/{len(train_loader)}, "
                        f"CTC: {avg_ctc_loss:.3f}, VQ: {avg_vq_loss:.3f}, "
                        f"Total: {avg_loss:.3f}"
                    )

                    writer.add_scalar("loss/ctc", avg_ctc_loss, steps)
                    writer.add_scalar("loss/vq", avg_vq_loss, steps)
                    writer.add_scalar("loss/total", avg_loss, steps)
                    writer.add_scalar("loss/vq_weight", vq_loss_weight, steps)

                    ctc_losses = []
                    vq_losses = []

        # --- Evaluation End of Epoch ---
        evaluate(model, val_loader, tokenizer, blank_token, device, epoch)
        
        # Save Checkpoint
        model.save(latest_path)

def evaluate(model, dataloader, tokenizer, blank_token, device, epoch):
    print(f"\n--- Evaluation Sample for Epoch {epoch+1} ---")
    model.eval()
    with torch.no_grad():
        try:
            # Fetch one batch from validation loader
            batch = next(iter(dataloader))
            audio = batch["audio"].to(device)
            # Use raw ground truth texts from the batch if available
            ground_truth_texts = batch["text"] 

            output, _ = model(audio) # (B, T', vocab)
            
            # Apply log_softmax if model doesn't (it does)
            # Decode
            predicted_texts = decode_ctc_output(output, tokenizer, blank_token)

            for i in range(min(5, len(ground_truth_texts))): # Print first 5 examples
                print(f"  Ground Truth: \"{ground_truth_texts[i]}\"")
                print(f"  Prediction:   \"{predicted_texts[i]}\"")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            
    print("------------------------------------------\n")
    model.train()

if __name__ == "__main__":
    train_model()
