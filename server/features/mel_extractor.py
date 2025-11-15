"""
Mel Spectrogram Feature Extractor for MVHST
Extracts mel spectrogram features from audio signals.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np


class MelSpectrogramExtractor(nn.Module):
    """Extracts mel spectrogram features from audio."""
    
    def __init__(self, 
                 sample_rate=22050,
                 n_fft=2048,
                 hop_length=512,
                 n_mels=128,
                 fmin=0.0,
                 fmax=None):
        """
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks
            fmin: Minimum frequency for mel filters
            fmax: Maximum frequency for mel filters (None = sample_rate/2)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        if fmax is None:
            fmax = sample_rate // 2
        
        # Create mel spectrogram transform
        # Use Spectrogram + MelScale for better compatibility across versions
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=fmin,
            f_max=fmax,
            n_stft=n_fft // 2 + 1
        )
        
    def forward(self, audio):
        """
        Args:
            audio: Tensor of shape (batch, samples) or (samples,)
        
        Returns:
            mel_spec: Tensor of shape (batch, n_mels, time) or (n_mels, time)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Compute spectrogram
        spec = self.spectrogram(audio)
        
        # Convert to mel scale
        mel_spec = self.mel_scale(spec)
        
        # Convert to log scale
        mel_spec = torch.log10(mel_spec + 1e-10)
        
        if audio.dim() == 1:
            mel_spec = mel_spec.squeeze(0)
        
        return mel_spec

