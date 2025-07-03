#!/usr/bin/env python3
"""
Test script to verify the data pipeline components work correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.make_dataset import AmazonBeautyDataProcessor
from utils.seed import set_seed
from utils.logging import setup_logger
from utils.serialization import save_json, load_json

def test_data_processor_init():
    """Test data processor initialization."""
    print("Testing data processor initialization...")
    
    processor = AmazonBeautyDataProcessor(
        raw_data_dir="./test_data/raw",
        processed_data_dir="./test_data/processed"
    )
    
    assert processor.raw_data_dir.exists()
    assert processor.processed_data_dir.exists()
    print("✓ Data processor initialized successfully")

def test_utilities():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print("✓ Seed set successfully")
    
    # Test logging
    logger = setup_logger("test_logger")
    logger.info("Test log message")
    print("✓ Logger created successfully")
    
    # Test serialization
    test_data = {"test": "data", "numbers": [1, 2, 3]}
    test_path = Path("./test_data/test_serialization.json")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_json(test_data, test_path)
    loaded_data = load_json(test_path)
    assert loaded_data == test_data
    
    # Cleanup
    test_path.unlink()
    print("✓ Serialization works correctly")

def test_directory_structure():
    """Test that all required directories exist."""
    print("Testing directory structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "models/teacher",
        "models/student",
        "train",
        "evaluate",
        "configs",
        "scripts",
        "utils"
    ]
    
    for dir_path in required_dirs:
        assert Path(dir_path).exists(), f"Directory {dir_path} does not exist"
    
    print("✓ All required directories exist")

def test_imports():
    """Test that all important modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        from datasets import load_dataset
        from transformers import AutoModel
        print("✓ All critical imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests."""
    print("Running data pipeline tests...\n")
    
    try:
        test_directory_structure()
        test_imports()
        test_utilities()
        test_data_processor_init()
        
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run data processing: python data/make_dataset.py")
        print("3. The system is ready for model implementation!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 