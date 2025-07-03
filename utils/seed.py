"""
Seed management utilities for reproducible experiments.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but more reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)


def get_seed_from_env(default: int = 42) -> int:
    """Get seed from environment variable or use default."""
    return int(os.environ.get('RANDOM_SEED', default)) 