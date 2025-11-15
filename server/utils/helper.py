"""
Helper utilities for MVHST training.
"""

import torch
import numpy as np
from pathlib import Path
import json


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', None)
    accuracy = checkpoint.get('accuracy', None)
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")
    
    return epoch, loss, accuracy


def calculate_accuracy(outputs, labels):
    """Calculate classification accuracy."""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def get_device():
    """Get available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_drone_folders(data_dir, drone_names):
    """Create empty drone folders for data organization."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    for drone_name in drone_names:
        drone_folder = data_path / drone_name
        drone_folder.mkdir(exist_ok=True)
        print(f"Created folder: {drone_folder}")


def save_class_mapping(class_to_idx, filepath):
    """Save class to index mapping."""
    with open(filepath, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Class mapping saved to {filepath}")


def load_class_mapping(filepath):
    """Load class to index mapping."""
    with open(filepath, 'r') as f:
        class_to_idx = json.load(f)
    return class_to_idx

