"""
Serialization utilities for saving and loading models and data.
"""

import pickle
import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """Save object to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_numpy(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save numpy array to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, array)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load numpy array from file."""
    return np.load(filepath)


def save_torch_model(model: torch.nn.Module, filepath: Union[str, Path]) -> None:
    """Save PyTorch model state dict."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_torch_model(model: torch.nn.Module, filepath: Union[str, Path]) -> torch.nn.Module:
    """Load PyTorch model state dict."""
    state_dict = torch.load(filepath, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Union[str, Path]
) -> None:
    """Save training checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint 