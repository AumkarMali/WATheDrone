"""
Transformer Encoder for MVHST
Processes spectral features using multi-head self-attention.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for processing spectral features."""
    
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_len=5000):
        """
        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(128, d_model)  # Assuming 128 frequency bins
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, freq_bins, time_frames)
        
        Returns:
            encoded: Tensor of shape (batch, time_frames, d_model)
        """
        batch_size, freq_bins, time_frames = x.shape
        
        # Transpose to (batch, time_frames, freq_bins)
        x = x.transpose(1, 2)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Project output
        encoded = self.output_projection(encoded)
        
        return encoded

