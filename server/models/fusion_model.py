"""
Multi-View Fusion Model for MVHST
Combines features from multiple views (mel, CQT, harmonic) using attention.
"""

import torch
import torch.nn as nn
from models.transformer_encoder import TransformerEncoder


class ViewAttention(nn.Module):
    """Attention mechanism for fusing multiple views."""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        
    def forward(self, views):
        """
        Args:
            views: List of tensors, each of shape (batch, time, d_model)
        
        Returns:
            fused: Tensor of shape (batch, time, d_model)
        """
        # Stack views: (n_views, batch, time, d_model)
        views_stack = torch.stack(views, dim=0)
        n_views, batch, time, d_model = views_stack.shape
        
        # Reshape to (batch, n_views, time, d_model)
        views_stack = views_stack.transpose(0, 1)
        
        # Compute attention
        queries = self.query(views_stack)  # (batch, n_views, time, d_model)
        keys = self.key(views_stack)
        values = self.value(views_stack)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, values)
        
        # Average across views
        fused = attended.mean(dim=1)  # (batch, time, d_model)
        
        return fused


class MVHSTModel(nn.Module):
    """Multi-View Harmonic Spectrum Transformer for drone classification."""
    
    def __init__(self,
                 num_classes,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 mel_freq_bins=128,
                 cqt_freq_bins=84,
                 harmonic_bins=5):
        """
        Args:
            num_classes: Number of drone classes
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            mel_freq_bins: Number of mel frequency bins
            cqt_freq_bins: Number of CQT frequency bins
            harmonic_bins: Number of harmonic bins
        """
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Separate encoders for each view
        self.mel_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.mel_encoder.input_projection = nn.Linear(mel_freq_bins, d_model)
        
        self.cqt_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.cqt_encoder.input_projection = nn.Linear(cqt_freq_bins, d_model)
        
        self.harmonic_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.harmonic_encoder.input_projection = nn.Linear(harmonic_bins, d_model)
        
        # View fusion
        self.view_attention = ViewAttention(d_model)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
    def forward(self, mel_features, cqt_features, harmonic_features):
        """
        Args:
            mel_features: Tensor of shape (batch, mel_freq_bins, time)
            cqt_features: Tensor of shape (batch, cqt_freq_bins, time)
            harmonic_features: Tensor of shape (batch, harmonic_bins, time)
        
        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        # Encode each view
        mel_encoded = self.mel_encoder(mel_features)  # (batch, time, d_model)
        cqt_encoded = self.cqt_encoder(cqt_features)
        harmonic_encoded = self.harmonic_encoder(harmonic_features)
        
        # Fuse views
        fused = self.view_attention([mel_encoded, cqt_encoded, harmonic_encoded])
        
        # Global pooling: (batch, time, d_model) -> (batch, d_model)
        fused = fused.transpose(1, 2)  # (batch, d_model, time)
        pooled = self.global_pool(fused).squeeze(-1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

