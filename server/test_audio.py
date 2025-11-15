"""
Test script to classify a single audio file using the MVHST model.
"""

import torch
import torchaudio
from pathlib import Path
import json
import numpy as np
import librosa

import config
from models.fusion_model import MVHSTModel
from features.mel_extractor import MelSpectrogramExtractor
from features.cqt_extractor import CQTExtractor
from features.harmonic_features import HarmonicFeatureExtractor
from utils.helper import get_device, load_class_mapping


def load_audio(filepath, sample_rate=22050, duration=5.0):
    """Load and preprocess audio file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Try different backends for loading audio
    waveform = None
    sr = None

    # Try torchaudio with soundfile backend first
    try:
        waveform, sr = torchaudio.load(str(filepath), backend='soundfile')
    except (RuntimeError, Exception):
        try:
            # Try with sox backend
            waveform, sr = torchaudio.load(str(filepath), backend='sox')
        except (RuntimeError, Exception):
            try:
                # Fallback to librosa (most reliable)
                audio_data, sr = librosa.load(str(filepath), sr=sample_rate, mono=True)
                # Convert to torch tensor: (samples,) -> (1, samples)
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            except Exception as e:
                raise RuntimeError(f"Failed to load audio file {filepath}: {str(e)}")

    # Convert to tensor if needed
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform)

    # Ensure waveform is float32
    waveform = waveform.float()

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Trim or pad to target length
    target_length = int(sample_rate * duration)
    current_length = waveform.shape[-1]

    if current_length > target_length:
        # Take center portion
        start = (current_length - target_length) // 2
        waveform = waveform[:, start:start + target_length]
    elif current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform.squeeze(0)  # Remove channel dimension


def predict_audio(model, feature_extractors, audio, device, class_mapping):
    """Predict drone class for audio."""
    model.eval()

    # Move audio to device
    audio = audio.to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Extract features
        print("Extracting features...")
        mel_features = feature_extractors['mel'](audio)
        cqt_features = feature_extractors['cqt'](audio)
        harmonic_features = feature_extractors['harmonic'](audio)

        print(f"Mel features shape: {mel_features.shape}")
        print(f"CQT features shape: {cqt_features.shape}")
        print(f"Harmonic features shape: {harmonic_features.shape}")

        # Forward pass
        print("Running inference...")
        logits = model(mel_features, cqt_features, harmonic_features)

        # Get predictions
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()

        # Get class name
        idx_to_class = {v: k for k, v in class_mapping.items()}
        predicted_class = idx_to_class[predicted_class_idx]

        # Get all predictions sorted by probability
        all_probs = probabilities[0].cpu().numpy()
        sorted_indices = np.argsort(all_probs)[::-1]  # Sort descending

        all_predictions = [
            {
                'class': idx_to_class[idx],
                'probability': all_probs[idx],
                'rank': rank + 1
            }
            for rank, idx in enumerate(sorted_indices)
        ]

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }


def main():
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load class mapping
    class_mapping_path = Path(config.LOG_DIR) / 'class_mapping.json'
    if not class_mapping_path.exists():
        print(f"Warning: Class mapping not found at {class_mapping_path}")
        print("Please train the model first or create a class mapping file.")
        return

    class_mapping = load_class_mapping(str(class_mapping_path))
    num_classes = len(class_mapping)
    print(f"Loaded {num_classes} classes: {list(class_mapping.keys())}")

    # Initialize feature extractors
    print("\nInitializing feature extractors...")
    feature_extractors = {
        'mel': MelSpectrogramExtractor(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.MEL_N_FFT,
            hop_length=config.MEL_HOP_LENGTH,
            n_mels=config.MEL_N_MELS,
            fmin=config.MEL_FMIN,
            fmax=config.MEL_FMAX
        ).to(device).eval(),
        'cqt': CQTExtractor(
            sample_rate=config.SAMPLE_RATE,
            hop_length=config.CQT_HOP_LENGTH,
            bins_per_octave=config.CQT_BINS_PER_OCTAVE,
            n_bins=config.CQT_N_BINS,
            fmin=config.CQT_FMIN
        ).to(device).eval(),
        'harmonic': HarmonicFeatureExtractor(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.HARMONIC_N_FFT,
            hop_length=config.HARMONIC_HOP_LENGTH,
            n_harmonics=config.HARMONIC_N_HARMONICS
        ).to(device).eval()
    }

    # Initialize model
    print("Initializing model...")
    model = MVHSTModel(
        num_classes=num_classes,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        mel_freq_bins=config.MEL_N_MELS,
        cqt_freq_bins=config.CQT_N_BINS,
        harmonic_bins=config.HARMONIC_N_HARMONICS
    ).to(device)

    # Load trained model if available
    checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pt'
    if checkpoint_path.exists():
        print(f"Loading trained model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()  # CRITICAL: Set to eval mode for inference
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Validation accuracy: {checkpoint.get('accuracy', 'N/A')}")
        print("Model set to evaluation mode")
    else:
        print("Warning: No trained model found. Using randomly initialized weights.")
        print("Predictions will not be meaningful without training.")
        model.eval()  # Set to eval mode even without trained weights

    # Load test audio
    # Try multiple possible locations
    audio_file = None
    possible_paths = [
        Path('Test16.wav'),
        Path(config.DATA_DIR) / 'Test16.wav',
        Path.cwd() / 'Test16.wav',
        Path(__file__).parent / 'Test16.wav'
    ]

    for path in possible_paths:
        if path.exists():
            audio_file = path
            break

    if audio_file is None:
        print("Error: Audio file 'Test16.wav' not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease place Test16.wav in the MV-HST directory or specify the path.")
        return

    print(f"\nLoading audio file: {audio_file}")
    audio = load_audio(audio_file, sample_rate=config.SAMPLE_RATE, duration=config.DURATION)
    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {audio.shape[0] / config.SAMPLE_RATE:.2f} seconds")

    # Make prediction
    print("\n" + "=" * 50)
    result = predict_audio(model, feature_extractors, audio, device, class_mapping)

    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll Predictions (sorted by probability):")
    print("-" * 50)
    for pred in result['all_predictions']:
        marker = ">>>" if pred['class'] == result['predicted_class'] else "   "
        print(f"{marker} {pred['rank']:2d}. {pred['class']:<12} {pred['probability']:>7.2%}")
    print("=" * 50)


if __name__ == '__main__':
    main()

