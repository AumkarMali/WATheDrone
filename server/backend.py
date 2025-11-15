from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from pydub import AudioSegment
import time
import gc
import torch
import soundfile as sf
import librosa
import numpy as np

app = Flask(__name__)
# Enable CORS for all routes - permissive for local development
# In production, you should restrict origins
CORS(app,
     origins="*",
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     supports_credentials=False,
     max_age=3600)

# MVHST Model loading - lazy load to avoid startup crashes
mvhst_model = None
feature_extractors = None
class_mapping = None
_model_loading = False
_model_load_error = None

# OpenUnmix model caching
umx_model = None
_umx_model_loading = False
_umx_model_load_error = None


def load_model():
    """Lazy load the MVHST drone detection model"""
    global mvhst_model, feature_extractors, class_mapping, _model_loading, _model_load_error

    if mvhst_model is not None and feature_extractors is not None:
        return True  # Already loaded

    if _model_loading:
        return False  # Currently loading

    if _model_load_error:
        return False  # Previous load failed

    try:
        _model_loading = True
        print("Loading MVHST model...")

        # Import here to avoid startup crashes
        from pathlib import Path
        import config
        from models.fusion_model import MVHSTModel
        from features.mel_extractor import MelSpectrogramExtractor
        from features.cqt_extractor import CQTExtractor
        from features.harmonic_features import HarmonicFeatureExtractor
        from utils.helper import get_device, load_class_mapping

        # Get device
        device = get_device()
        print(f"Using device: {device}")

        # Load class mapping
        class_mapping_path = Path(config.LOG_DIR) / 'class_mapping.json'
        if not class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping not found at {class_mapping_path}. Please train the model first.")

        class_mapping = load_class_mapping(str(class_mapping_path))
        num_classes = len(class_mapping)
        print(f"Loaded {num_classes} classes: {list(class_mapping.keys())}")

        # Initialize feature extractors
        print("Initializing feature extractors...")
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
        print("Initializing MVHST model...")
        mvhst_model = MVHSTModel(
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

        # Load trained model
        checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pt'
        if checkpoint_path.exists():
            print(f"Loading trained model from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            mvhst_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            mvhst_model.eval()
            print(f"Model loaded from epoch {checkpoint['epoch']}")
        else:
            print("Warning: No trained model found. Using randomly initialized weights.")
            mvhst_model.eval()

        # Disable gradient computation globally
        for param in mvhst_model.parameters():
            param.requires_grad = False

        _model_loading = False
        print("MVHST model loaded successfully!")
        return True

    except Exception as e:
        error_msg = f"Error loading MVHST model: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        _model_load_error = error_msg
        _model_loading = False
        mvhst_model = None
        feature_extractors = None
        class_mapping = None
        return False


# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac', 'mpeg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_delete_file(filepath, max_retries=10, retry_delay=0.2):
    """Safely delete a file with retry logic for Windows file locking issues"""
    if not filepath or not os.path.exists(filepath):
        return True  # File doesn't exist, nothing to delete

    for attempt in range(max_retries):
        try:
            os.remove(filepath)
            return True
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                # Exponential backoff with longer delays
                delay = retry_delay * (2 ** attempt)  # Exponential: 0.2s, 0.4s, 0.8s, 1.6s, etc.
                time.sleep(delay)
                gc.collect()  # Force garbage collection to release file handles
            else:
                print(f"Warning: Could not delete file {filepath} after {max_retries} attempts: {e}")
                return False
    return True


def predict_drone(audio_path):
    """Predict drone class using MVHST model"""
    global _model_load_error, mvhst_model, feature_extractors, class_mapping

    # Try to load model if not loaded
    if not load_model():
        error_msg = _model_load_error or "Model not loaded"
        return None, None, None, error_msg

    try:
        import torchaudio
        import torch
        import gc
        from pathlib import Path
        import config
        from utils.helper import get_device

        device = get_device()

        # Load audio file with fallback backends
        waveform = None
        sr = None
        try:
            waveform, sr = torchaudio.load(str(audio_path), backend='soundfile')
        except (RuntimeError, Exception):
            try:
                waveform, sr = torchaudio.load(str(audio_path), backend='sox')
            except (RuntimeError, Exception):
                # Fallback to librosa
                audio_data, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
                waveform = torch.from_numpy(audio_data).unsqueeze(0)

        if waveform is None:
            raise RuntimeError("Failed to load audio file with any backend")

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Trim or pad to target length
        target_length = int(config.SAMPLE_RATE * config.DURATION)
        current_length = waveform.shape[-1]

        if current_length > target_length:
            # Take center portion
            start = (current_length - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Move to device and ensure batch dimension
        audio = waveform.to(device)  # Keep as (1, samples) for batch processing

        # Extract features
        with torch.no_grad():
            mvhst_model.eval()
            mel_features = feature_extractors['mel'](audio)
            cqt_features = feature_extractors['cqt'](audio)
            harmonic_features = feature_extractors['harmonic'](audio)

            # Ensure all features have batch dimension (batch, freq_bins, time)
            if mel_features.dim() == 2:
                mel_features = mel_features.unsqueeze(0)
            if cqt_features.dim() == 2:
                cqt_features = cqt_features.unsqueeze(0)
            if harmonic_features.dim() == 2:
                harmonic_features = harmonic_features.unsqueeze(0)

            # Debug: print shapes to verify
            print(
                f"Feature shapes - Mel: {mel_features.shape}, CQT: {cqt_features.shape}, Harmonic: {harmonic_features.shape}")

            # Forward pass
            logits = mvhst_model(mel_features, cqt_features, harmonic_features)

            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=1)[0]

            # Get predicted class
            predicted_class_idx = torch.argmax(probabilities, dim=0).item()
            confidence = float(probabilities[predicted_class_idx].item())

            # Get class name
            idx_to_class = {v: k for k, v in class_mapping.items()}
            predicted_class = idx_to_class[predicted_class_idx]

            # Get all probabilities for all classes (as percentages 0-100)
            all_probabilities = {}
            prob_list = []
            for class_id in range(len(class_mapping)):
                class_name = idx_to_class[class_id]
                # Convert to percentage (0-100)
                prob_value = float(probabilities[class_id].item()) * 100.0
                all_probabilities[class_name] = round(prob_value, 2)
                prob_list.append((class_name, round(prob_value, 2)))
                # Also add short form (A, B, C, etc.) for frontend compatibility
                if class_name.startswith('drone_'):
                    short_name = class_name.replace('drone_', '')
                    all_probabilities[short_name] = round(prob_value, 2)

            # Print all predictions sorted by probability
            print("\n" + "=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print("\nAll Predictions (sorted by probability):")
            print("-" * 60)
            # Sort by probability (descending)
            prob_list.sort(key=lambda x: x[1], reverse=True)
            for rank, (class_name, prob) in enumerate(prob_list, 1):
                marker = ">>>" if class_name == predicted_class else "   "
                print(f"{marker} {rank:2d}. {class_name:<15} {prob:>6.2f}%")
            print("=" * 60 + "\n")

        # Clear cache and free memory
        del audio, mel_features, cqt_features, harmonic_features, logits, probabilities, waveform
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return predicted_class, confidence, all_probabilities, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clean up on error
        import gc
        gc.collect()
        return None, None, None, str(e)


@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    """Health check endpoint"""
    global mvhst_model, feature_extractors, _model_load_error
    model_status = "loaded" if (mvhst_model is not None and feature_extractors is not None) else "not loaded"
    if _model_load_error:
        model_status = f"error: {_model_load_error[:100]}"  # Truncate long errors

    return jsonify({
        'status': 'success',
        'message': 'MVHST Drone Detection Server is running',
        'model_status': model_status,
        'timestamp': datetime.utcnow().isoformat(),
        'cors_enabled': True
    })


@app.route('/test', methods=['GET', 'OPTIONS'])
def test_connection():
    """Simple test endpoint to verify frontend can connect"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        return response

    return jsonify({
        'status': 'success',
        'message': 'Connection test successful',
        'server': 'Flask backend',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.before_request
def log_request_info():
    """Log request details for debugging"""
    print(f"\n{'=' * 60}")
    print(f"Request: {request.method} {request.path}")
    print(f"Origin: {request.headers.get('Origin', 'N/A')}")
    print(f"Content-Type: {request.headers.get('Content-Type', 'N/A')}")
    print(f"URL: {request.url}")
    if request.method == 'OPTIONS':
        print("This is a CORS preflight request")
    print(f"{'=' * 60}\n")


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle audio file upload and convert to WAV"""
    try:
        # Check if file is present in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['audio']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed formats: WAV, MP3, M4A, AAC, OGG, FLAC'
            }), 400

        # Generate unique filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        temp_filename = f"{uuid.uuid4()}.{file_extension}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        # Save temporarily
        file.save(temp_filepath)

        try:
            # Convert to WAV format (standard for ML/Audio processing)
            wav_filename = f"{uuid.uuid4()}.wav"
            wav_filepath = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)

            # Load audio file
            audio = AudioSegment.from_file(temp_filepath)

            # Convert to WAV: 22050Hz (MVHST sample rate), mono channel
            import config as mv_config
            audio.set_frame_rate(mv_config.SAMPLE_RATE).set_channels(1).export(wav_filepath, format="wav")

            # Explicitly delete audio object to release file handles
            del audio
            gc.collect()

            # Add delay to allow Windows to release file handles from ffmpeg/ffprobe
            time.sleep(0.5)

            # Get file size
            file_size = os.path.getsize(wav_filepath)

            # Run drone detection prediction
            prediction_label = None
            prediction_confidence = None
            prediction_error = None

            try:
                prediction_label, prediction_confidence, all_probabilities, prediction_error = predict_drone(
                    wav_filepath)
            except Exception as pred_error:
                print(f"Prediction error: {str(pred_error)}")
                prediction_error = str(pred_error)
                all_probabilities = None

            # Clean up original file using safe delete
            safe_delete_file(temp_filepath)

            # Build response
            response_data = {
                'status': 'success',
                'message': 'File uploaded and converted successfully',
                'filename': original_filename,
                'saved_as': wav_filename,
                'wav_path': wav_filepath,
                'size': file_size,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Add prediction data if available
            if prediction_label is not None and prediction_confidence is not None:
                response_data['prediction'] = {
                    'label': prediction_label,
                    'confidence': round(prediction_confidence * 100, 2)  # Convert to percentage
                }
                # Add all probabilities if available (already in percentage format from predict_drone)
                if all_probabilities is not None:
                    response_data['prediction']['probabilities'] = all_probabilities
            elif prediction_error:
                response_data['prediction_error'] = prediction_error

            return jsonify(response_data), 200

        except Exception as conversion_error:
            # Clean up on conversion error
            safe_delete_file(temp_filepath)
            raise conversion_error

    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


def load_umx_model():
    """Lazy load the OpenUnmix model and cache it globally"""
    global umx_model, _umx_model_loading, _umx_model_load_error

    if umx_model is not None:
        return True  # Already loaded

    if _umx_model_loading:
        return False  # Currently loading

    if _umx_model_load_error:
        return False  # Previous load failed

    try:
        _umx_model_loading = True

        # Import here to avoid startup crashes
        import openunmix

        # Load UMX model
        umx_model = openunmix.umxhq(pretrained=True).eval()

        # Move to CPU and disable gradients
        umx_model = umx_model.cpu()
        for param in umx_model.parameters():
            param.requires_grad = False

        _umx_model_loading = False
        print("OpenUnmix model loaded successfully")
        return True

    except ImportError:
        error_msg = "openunmix is not installed. Please install it with: pip install openunmix"
        print(error_msg)
        _umx_model_load_error = error_msg
        _umx_model_loading = False
        umx_model = None
        return False
    except Exception as e:
        error_msg = f"Error loading OpenUnmix model: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        _umx_model_load_error = error_msg
        _umx_model_loading = False
        umx_model = None
        return False


@app.route('/denoise', methods=['POST'])
def denoise_audio():
    """Denoise audio using openunmix to extract drone sounds"""
    global umx_model, _umx_model_load_error

    try:
        # Check if file is present in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['audio']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed formats: WAV, MP3, M4A, AAC, OGG, FLAC'
            }), 400

        # Generate unique filename for input
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        temp_filename = f"{uuid.uuid4()}.{file_extension}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        # Save temporarily
        file.save(temp_filepath)

        try:
            # Try to load model if not loaded
            if not load_umx_model():
                error_msg = _umx_model_load_error or "OpenUnmix model not loaded"
                safe_delete_file(temp_filepath)
                return jsonify({
                    'error': f'Model loading failed: {error_msg}'
                }), 500

            # Generate output filename
            output_filename = f"{uuid.uuid4()}_denoised.wav"
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            # ------------------------
            # Load mono audio
            # ------------------------
            y, sr_old = librosa.load(temp_filepath, sr=None, mono=True)

            # ------------------------
            # Resample to 44.1 kHz for UMX
            # ------------------------
            if sr_old != 44100:
                y = librosa.resample(y, orig_sr=sr_old, target_sr=44100)
            sr = 44100

            # ------------------------
            # UMX requires stereo → duplicate mono channel
            # ------------------------
            y_stereo = np.stack([y, y], axis=0).astype(np.float32)  # (2, T)

            # Shape for model: (batch=1, channels=2, samples=T)
            audio_tensor = torch.tensor(y_stereo)[None, :, :]

            # ------------------------
            # Run separation
            # ------------------------
            print(f"Running separation on audio tensor shape: {audio_tensor.shape}")
            with torch.no_grad():
                separated = umx_model(audio_tensor)

            print("UMX output shape:", separated.shape)
            # expected: (1, 4, 2, T)

            # ------------------------
            # Extract "OTHER" stem (index 3)
            # ------------------------
            other = separated[0, 3].cpu().numpy()  # shape (2, T)

            # Convert stereo to mono
            other_mono = other.mean(axis=0)

            # ------------------------
            # Match loudness to original audio
            # ------------------------
            orig_rms = np.sqrt(np.mean(y ** 2))
            out_rms = np.sqrt(np.mean(other_mono ** 2))
            if out_rms > 0:
                gain = orig_rms / out_rms
                other_mono = other_mono * gain

            # ------------------------
            # Save
            # ------------------------
            sf.write(output_filepath, other_mono, sr)
            print("Saved:", output_filepath)

            # Clean up temporary input file
            safe_delete_file(temp_filepath)

            # Clean up tensors to free memory
            del y, y_stereo, audio_tensor, separated, other, other_mono
            gc.collect()

            # Return JSON with file URL - frontend will fetch the file using this URL
            return jsonify({
                'status': 'success',
                'message': 'Audio denoised successfully',
                'filename': output_filename,
                'url': f'/uploads/{output_filename}'
            }), 200

        except Exception as denoise_error:
            # Clean up on error
            safe_delete_file(temp_filepath)
            error_msg = str(denoise_error)
            print(f"Denoising error: {error_msg}")
            import traceback
            traceback.print_exc()
            # Return more detailed error message
            return jsonify({
                'error': f'Denoising failed: {error_msg}'
            }), 500

    except Exception as e:
        print(f"Error in denoise endpoint: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/uploads/<filename>', methods=['GET'])
def serve_upload(filename):
    """Serve uploaded/processed audio files"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            return send_file(filepath, mimetype='audio/wav')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    files.append({
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        ).isoformat()
                    })

        return jsonify({
            'status': 'success',
            'files': files,
            'count': len(files)
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'=' * 60}")
    print("Flask Server Starting...")
    print(f"Available routes:")
    print(f"  GET  / - Health check")
    print(f"  POST /upload - Upload audio file")
    print(f"  POST /denoise - Denoise audio using openunmix")
    print(f"  GET  /files - List uploaded files")
    print(f"  GET  /uploads/<filename> - Serve audio files")
    print(f"{'=' * 60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)

# For cloud platforms that need PORT from environment
# The Procfile/gunicorn will handle this automatically
