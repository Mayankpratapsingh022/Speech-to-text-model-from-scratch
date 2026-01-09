import torch
import torch.nn as nn
import config

from downsampling import DownsamplingNetwork
from transformer import TransformerEncoder
from rvq import ResidualVectorQuantizer

class TranscribeModel(nn.Module):
    def __init__(
        self,
        num_codebooks: int = config.NUM_CODEBOOKS,
        codebook_size: int = config.CODEBOOK_SIZE,
        embedding_dim: int = config.EMBEDDING_DIM,
        vocab_size: int = 50257+100, # Approximate for GPT2 + special tokens. 
        # Ideally, vocab_size should be dynamic or passed from tokenizer.
        # We will default to a safely large number or update init to require it.
        # For now, let's allow it to be passed or default to config if present.
        strides: list[int] = config.STRIDES,
        initial_mean_pooling_kernel_size: int = config.INITIAL_POOLING_KERNEL,
        num_transformer_layers: int = config.NUM_LAYERS,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        num_heads: int = config.NUM_HEADS,
    ):
        super().__init__()

        # store for saving/loading
        self.options = {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "embedding_dim": embedding_dim,
            "vocab_size": vocab_size,
            "strides": strides,
            "num_transformer_layers": num_transformer_layers,
            "initial_mean_pooling_kernel_size": initial_mean_pooling_kernel_size,
            "max_seq_length": max_seq_length,
            "num_heads": num_heads
        }

        # 1) Conv front‑end
        self.downsampling_network = DownsamplingNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim // 2,
            in_channels=1,
            initial_mean_pooling_kernel_size=initial_mean_pooling_kernel_size,
            strides=strides,
        )

        # 2) Transformer encoder before RVQ
        self.pre_rvq_transformer = TransformerEncoder(
            embed_size=embedding_dim,
            num_layers=num_transformer_layers,
            max_seq_length=max_seq_length,
            num_heads=num_heads,
        )

        # 3) Residual vector quantizer over encoder features
        self.rvq = ResidualVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
        )

        # 4) Output projection to vocab
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, time) raw audio waveform (already normalized / resampled)
        returns:
            log_probs: (batch, T', vocab_size)
            vq_loss: scalar auxiliary loss from RVQ
        """
        # Ensure input is on correct device if not already (or let pytorch handle mismatch error)
        # x shape check
        if x.dim() == 2:
            # add channel dimension for Conv1d: (B, 1, T)
            x = x.unsqueeze(1)
        
        # conv downsampling: (B, T_conv, D)
        # Output is (B, T', D) because DownsamplingNetwork handles transpose
        x = self.downsampling_network(x)

        # transformer encoder: (B, T_enc, D)
        x = self.pre_rvq_transformer(x)

        # RVQ over encoder features
        x, vq_loss = self.rvq(x)

        # project to vocabulary and log‑softmax for CTC / seq‑loss
        x = self.output_layer(x)          # (B, T_enc, vocab)
        
        # Use log_softmax for CTC compatibility
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)  

        return log_probs, vq_loss

    # --------- saving / loading helpers ---------

    def save(self, path: str):
        print("Saving model to", path)
        torch.save({"model": self.state_dict(), "options": self.options}, path)

    @staticmethod
    def load(path: str, map_location: str | torch.device | None = None):
        print("Loading model from", path)
        checkpoint = torch.load(path, map_location=map_location)
        model = TranscribeModel(**checkpoint["options"])
        model.load_state_dict(checkpoint["model"])
        return model


