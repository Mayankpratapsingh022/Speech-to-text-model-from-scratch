import torch
import torch.nn as nn
from vector_quantizer import VectorQuantizer
import config

class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ).
    Applies multiple VQ layers sequentially. Each subsequent layer quantizes the 
    residual error of the previous layer.
    """
    def __init__(self, 
                 num_codebooks: int = config.NUM_CODEBOOKS, 
                 codebook_size: int = config.CODEBOOK_SIZE, 
                 embedding_dim: int = config.EMBEDDING_DIM, 
                 commitment_cost: float = config.COMMITMENT_COST):
        super().__init__()
        self.codebooks = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, embedding_dim)
        returns:
          out: (batch, seq_len, embedding_dim)  # sum of all quantized residuals
          total_loss: scalar (sum of VQ losses over codebooks)
        """
        residual = x
        out = torch.zeros_like(x)
        total_loss = 0.0

        # We keep track of individual code indices if needed later (e.g. for EnCodec style tokens)
        # For now, just return reconstructed vector and loss as requested.

        for codebook in self.codebooks:
            # Quantize the current residual
            quantized_diff, loss = codebook(residual)
            
            # Accumulate the reconstruction
            out = out + quantized_diff
            
            # Update residual: subtract what we just quantized
            residual = residual - quantized_diff
            
            # Sum up losses from each stage
            total_loss = total_loss + loss

        return out, total_loss


