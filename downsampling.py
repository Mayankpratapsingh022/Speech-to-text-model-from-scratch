import torch
import torch.nn as nn
import config

class ResidualDownSampleBlock(nn.Module):
    """
    Residual block that performs downsampling.
    Has two convolution layers. The first preserves length, the second downsamples.
    Includes a skip connection, with a projection if dimensions change.
    """
    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()

        # First conv: preserves temporal length
        # Manual padding for 'same' roughly: (kernel_size - 1) // 2 for odd, or explicit for even.
        # Kernel size is 8 (even). padding='same' adds padding on both sides.
        # For kernel=8, padding=3 on left, 4 on right? Or 3.5? PyTorch same padding for even kernel might be asymmetric.
        # Let's use simple symmetric padding which is supported everywhere: padding = kernel_size // 2
        # But this changes output size slightly if not careful.
        # Ideally, we stick to odd kernels or accept size change.
        # User config has kernel=8.
        # Let's try standard padding and crop if needed, or just standard padding.
        # padding = kernel_size // 2 -> output size = L + 2*P - K + 1 = L + 8 - 8 + 1 = L + 1. Close enough.
        # Actually, let's just use padding = (kernel_size - 1) // 2 
        # For K=8, P=3. L_out = L + 6 - 8 + 1 = L - 1. 
        # We can pad to match input size manually, or just use PyTorch 2.1+ same padding?
        # The error was about channel size limit which is weird. It might be an internal lowering bug for 'same'.
        # Let's switch to padding = (kernel_size - 1) // 2 and see.
        padding_val = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_val,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv: downsamples via stride
        # We need padding calculation to match "same" behavior roughly if we want controlled output
        # But standard stride reduces length by factor 'stride'.
        # For simplicity and matching snippet, we rely on default padding=0 or explicit.
        # Snippet used default padding (0) for stride layer usually, but let's check.
        # If we want exact downsampling ratio, we might need padding.
        # Let's use padding to keep it consistent with "kernel//2" logic often used.
        # The user snippet didn't specify padding for conv2, so it would lose some border pixels.
        # Let's add padding to maintain cleaner shapes: padding = (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding 
        )
        
        self.relu = nn.ReLU()
        
        # Projection for residual connection
        # 1. If in_channels != out_channels, we need to project channels.
        # 2. Since conv2 downsamples time by 'stride', we must also downsample x by 'stride'.
        # We can use a 1x1 conv with stride for this.
        if in_channels != out_channels or stride != 1:
            self.residual_proj = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=stride,
                padding=0
            )
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # x: (batch, channels, time)
        
        # Residual path
        residual = self.residual_proj(x)
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        
        # Match dimensions if slight mismatches due to padding calculations (common in strided convs)
        if out.shape[-1] != residual.shape[-1]:
            # Crop to match the smaller one (usually out is slightly different if kernel/stride math isn't perfect)
            min_len = min(out.shape[-1], residual.shape[-1])
            out = out[..., :min_len]
            residual = residual[..., :min_len]
            
        out = out + residual
        return self.relu(out)

class DownsamplingNetwork(nn.Module):
    """
    Full downsampling network consisting of a pooling layer and stacked residual blocks.
    Projects raw audio to transformer embedding dimension.
    """
    def __init__(
        self,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        in_channels=config.INPUT_CHANNELS,
        initial_mean_pooling_kernel_size=config.INITIAL_POOLING_KERNEL,
        strides=config.STRIDES,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # Initial pooling to reduce length quickly
        self.mean_pooling = nn.AvgPool1d(
            kernel_size=initial_mean_pooling_kernel_size,
            stride=initial_mean_pooling_kernel_size 
        )

        # Stack of residual downsampling blocks
        current_in_channels = in_channels
        for i, s in enumerate(strides):
            # First block projects from input to hidden, subsequent stay hidden
            block_in = current_in_channels if i == 0 else hidden_dim
            
            self.layers.append(
                ResidualDownSampleBlock(
                    in_channels=block_in,
                    out_channels=hidden_dim,
                    stride=s,
                    kernel_size=config.KERNEL_SIZE,
                )
            )
            # After first block, we are at hidden_dim
        
        # Final projection to embedding_dim
        self.final_conv = nn.Conv1d(
            hidden_dim,
            embedding_dim,
            kernel_size=4,
            padding="same",
        )

    def forward(self, x):
        # x: (batch, 1, time)
        x = self.mean_pooling(x)
        
        for layer in self.layers:
            x = layer(x)

        x = self.final_conv(x)  # (B, embedding_dim, T')
        x = x.transpose(1, 2)   # (B, T', embedding_dim) for the transformer
        return x


