"""
Constant-Q Transform (CQT) Feature Extractor for MVHST
Extracts CQT features from audio signals using GPU-accelerated operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CQTExtractor(nn.Module):
    """Extracts Constant-Q Transform features from audio using GPU-accelerated operations."""
    
    def __init__(self,
                 sample_rate=22050,
                 hop_length=512,
                 bins_per_octave=12,
                 n_bins=84,
                 fmin=27.5):
        """
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for CQT
            bins_per_octave: Number of bins per octave
            n_bins: Total number of frequency bins
            fmin: Minimum frequency (Hz)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.n_bins = n_bins
        self.fmin = fmin
        
        # Pre-compute frequencies (will be moved to device in forward)
        self.register_buffer('frequencies', 
                            fmin * (2 ** (torch.arange(n_bins, dtype=torch.float32) / bins_per_octave)))
        
    def forward(self, audio):
        """
        Args:
            audio: Tensor of shape (batch, samples) or (samples,)
        
        Returns:
            cqt: Tensor of shape (batch, n_bins, time) or (n_bins, time)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, n_samples = audio.shape
        device = audio.device
        
        # Ensure frequencies are on the same device
        frequencies = self.frequencies.to(device)
        
        # Use STFT as base, then aggregate to CQT bins
        # This is a simplified but GPU-efficient CQT approximation
        n_fft = 2048  # Use fixed FFT size for efficiency
        window = torch.hann_window(n_fft, device=device)
        stft = torch.stft(
            audio.squeeze(0) if batch_size == 1 else audio.view(-1, n_samples),
            n_fft=n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        
        # Get magnitude spectrogram: (batch, freq, time)
        magnitude = torch.abs(stft)
        
        if batch_size == 1:
            magnitude = magnitude.unsqueeze(0)
        else:
            magnitude = magnitude.view(batch_size, -1, magnitude.shape[-1])
        
        # Map STFT bins to CQT bins
        # STFT frequency resolution
        stft_freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sample_rate, device=device)
        
        # Aggregate STFT bins to CQT bins
        cqt_list = []
        for freq in frequencies:
            # Find STFT bins near this CQT frequency
            # Use a simple window around the target frequency
            freq_diff = torch.abs(stft_freqs - freq)
            weights = torch.exp(-freq_diff / (freq * 0.1))  # Gaussian-like weighting
            
            # Weighted sum of STFT bins
            weighted = (magnitude * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
            cqt_list.append(weighted)
        
        # Stack: (batch, n_bins, time)
        cqt = torch.stack(cqt_list, dim=1)
        
        # Convert to log scale
        cqt = torch.log10(cqt + 1e-10)
        
        if squeeze_output:
            cqt = cqt.squeeze(0)
        
        return cqt

