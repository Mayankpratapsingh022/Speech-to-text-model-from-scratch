import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

class FeedForward(nn.Module):
    def __init__(self, embed_size: int, ff_hidden_mult: int = config.FF_HIDDEN_MULT, dropout: float = config.DROPOUT):
        super().__init__()
        hidden = ff_hidden_mult * embed_size
        self.layer1 = nn.Linear(embed_size, hidden)
        self.layer2 = nn.Linear(hidden, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, dropout: float = config.DROPOUT):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        # batch_first=True so we keep (batch, seq, embed) everywhere
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_size)

        self.ff = FeedForward(embed_size, dropout=dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        x: (batch, seq_len, embed_size)
        attn_mask: optional mask of shape (batch, seq_len) or (seq_len, seq_len)
        """
        # Multi‑head self‑attention
        # Note: key_padding_mask is (N, S) where True means PAD (ignore).
        # attn_mask usually refers to causal mask (S, S). 
        # For STT Encoder, usually we just need key_padding_mask to ignore padded audio frames.
        # The argument name here 'attn_mask' in signature is generic.
        # For torch.nn.MultiheadAttention: 
        # - key_padding_mask: (N, S), True for ignored positions.
        # - attn_mask: (S, S) or (N*num_heads, S, S), for causal masking.
        # We usually pass 'key_padding_mask' for padding.
        # Let's assume input 'attn_mask' acts as 'key_padding_mask' if shape matches (B, S).
        
        # We'll map the input argument appropriately based on usage in training loop.
        # For now, let's assume the user passes padding mask as 'key_padding_mask' kwarg or we adapt.
        # The snippet had: key_padding_mask=attn_mask. Let's keep that.
        
        attn_out, attn_weights = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=attn_mask,   
            need_weights=True,
        )

        # Residual + norm
        x = self.attn_norm(x + self.attn_dropout(attn_out))

        # Feed‑forward block with residual + norm
        ff_out = self.ff(x)
        x = self.ff_norm(x + self.ff_dropout(ff_out))

        return x, attn_weights

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embed_size: int, max_seq_length: int = config.MAX_SEQ_LENGTH):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)         # (T, 1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )                                                             # (E/2,)
        pe = torch.zeros(max_seq_length, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_embedding", pe)             # (T, E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_size)
        seq_len = x.size(1)
        # Ensure seq_len fits in max_seq_length
        if seq_len > self.positional_embedding.size(0):
             # Dynamically extend or error out. For now, crop or warn.
             # STT sequences can be long.
             pass
        return x + self.positional_embedding[:seq_len, :]

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        embed_size: int = config.EMBEDDING_DIM, 
        num_layers: int = config.NUM_LAYERS, 
        num_heads: int = config.NUM_HEADS, 
        max_seq_length: int = config.MAX_SEQ_LENGTH
    ):
        super().__init__()
        self.positional_encoding = SinusoidalPositionEncoding(embed_size, max_seq_length)
        self.transformer_blocks = nn.ModuleList(
            [SelfAttentionLayer(embed_size, num_heads) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        x: (batch, seq_len, embed_size) from the DownsamplingNetwork
        attn_mask: (batch, seq_len) boolean mask where True = PAD
        """
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x, _ = block(x, attn_mask=attn_mask)
        return x


