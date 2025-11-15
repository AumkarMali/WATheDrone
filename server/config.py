"""
Configuration file for MVHST training.
"""

# Data configuration
DATA_DIR = 'data'
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Feature extraction configuration
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128
MEL_FMIN = 0.0
MEL_FMAX = None  # None = sample_rate / 2

CQT_HOP_LENGTH = 512
CQT_BINS_PER_OCTAVE = 12
CQT_N_BINS = 84
CQT_FMIN = 27.5

HARMONIC_N_FFT = 2048
HARMONIC_HOP_LENGTH = 512
HARMONIC_N_HARMONICS = 5

# Model configuration
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# Augmentation configuration
TIME_SHIFT_PROB = 0.5
TIME_STRETCH_PROB = 0.3 
PITCH_SHIFT_PROB = 0.3
ADD_NOISE_PROB = 0.3
NOISE_LEVEL = 0.01

# Checkpoint and logging
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
SAVE_EVERY = 10  # Save checkpoint every N epochs
PRINT_EVERY = 10  # Print training info every N batches

# Device
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# Random seed
SEED = 42

