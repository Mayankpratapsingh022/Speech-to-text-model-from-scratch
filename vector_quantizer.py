import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer Module.
    Encodes input vectors to the nearest codebook vector.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: N x D
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize uniformly
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, embedding_dim)
        returns:
          quantized: (batch, seq_len, embedding_dim)
          loss: scalar VQ loss
        """
        batch_size, seq_len, embed_dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, embed_dim)         # (B*T, D)

        # Distances between each vector and all codebook entries
        # (x - e)^2 = x^2 + e^2 - 2xe
        # torch.cdist creates the Euclidean distance matrix
        distances = torch.cdist(flat_x, self.embedding.weight, p=2) # (B*T, N)

        # Nearest code index per vector
        encoding_indices = torch.argmin(distances, dim=1)           # (B*T,)

        # Look up quantized vectors and reshape back
        quantized = self.embedding(encoding_indices).view(
            batch_size, seq_len, embed_dim
        )

        # VQ-VAE loss: 
        # 1. Commitment loss: move x closer to chosen code (scale by commitment_cost)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        # 2. Codebook loss: move chosen code closer to x
        q_latent_loss = F.mse_loss(quantized, x.detach())
        
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        # We want gradients to flow from decoder to encoder, essentially bypassing the discrete sampling step
        quantized = x + (quantized - x).detach()

        return quantized, loss


