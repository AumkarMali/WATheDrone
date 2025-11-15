"""
Harmonic Feature Extractor for MVHST
Extracts harmonic and percussive components from audio.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np


class HarmonicFeatureExtractor(nn.Module):
    """Extracts harmonic features from audio signals."""
    
    def __init__(self,
                 sample_rate=22050,
                 n_fft=2048,
                 hop_length=512,
                 n_harmonics=5):
        """
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_harmonics: Number of harmonic components to extract
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_harmonics = n_harmonics
        
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,  # Return complex spectrogram
            window_fn=torch.hann_window
        )
        
    def _extract_harmonics(self, magnitude_spec):
        """
        Extract harmonic components from magnitude spectrogram.
        
        Args:
            magnitude_spec: Tensor of shape (batch, freq, time) or (freq, time)
        
        Returns:
            harmonic_features: Tensor of shape (batch, n_harmonics, time) or (n_harmonics, time)
        """
        if magnitude_spec.dim() == 2:
            magnitude_spec = magnitude_spec.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, n_freq, n_time = magnitude_spec.shape
        
        # Extract fundamental frequency (F0) estimate using peak picking
        # Simplified: use the frequency bin with maximum energy in each time frame
        f0_bins = torch.argmax(magnitude_spec, dim=1)  # (batch, time)
        
        # Extract harmonic components
        harmonic_features = []
        for h in range(1, self.n_harmonics + 1):
            # For each harmonic, extract energy at h * f0
            harmonic_energy = []
            for b in range(batch_size):
                frame_energies = []
                for t in range(n_time):
                    f0_bin = f0_bins[b, t].item()
                    harmonic_bin = min(int(f0_bin * h), n_freq - 1)
                    energy = magnitude_spec[b, harmonic_bin, t]
                    frame_energies.append(energy)
                harmonic_energy.append(torch.stack(frame_energies))
            
            harmonic_features.append(torch.stack(harmonic_energy, dim=0))
        
        harmonic_features = torch.stack(harmonic_features, dim=1)  # (batch, n_harmonics, time)
        
        # Convert to log scale
        harmonic_features = torch.log10(harmonic_features + 1e-10)
        
        if squeeze_output:
            harmonic_features = harmonic_features.squeeze(0)
        
        return harmonic_features
    
    def forward(self, audio):
        """
        Args:
            audio: Tensor of shape (batch, samples) or (samples,)
        
        Returns:
            harmonic_features: Tensor of shape (batch, n_harmonics, time) or (n_harmonics, time)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Compute STFT
        stft = self.stft(audio)
        
        # Get magnitude spectrogram
        magnitude_spec = torch.abs(stft)
        
        # Extract harmonic features
        harmonic_features = self._extract_harmonics(magnitude_spec)
        
        if audio.dim() == 1:
            harmonic_features = harmonic_features.squeeze(0)
        
        return harmonic_features

