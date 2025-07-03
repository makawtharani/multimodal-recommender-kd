#!/usr/bin/env python3
"""
Simple test script to verify basic project structure.
"""

import sys
import os
from pathlib import Path

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
        if not Path(dir_path).exists():
            print(f"✗ Directory {dir_path} does not exist")
            return False
    
    print("✓ All required directories exist")
    return True

def test_required_files():
    """Test that required files exist."""
    print("Testing required files...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        ".env.example",
        "data/make_dataset.py",
        "utils/seed.py",
        "utils/logging.py",
        "utils/serialization.py",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"✗ File {file_path} does not exist")
            return False
    
    print("✓ All required files exist")
    return True

def test_file_permissions():
    """Test that scripts have correct permissions."""
    print("Testing file permissions...")
    
    executable_files = [
        "data/make_dataset.py",
        "scripts/test_data_pipeline.py",
        "scripts/test_structure.py"
    ]
    
    for file_path in executable_files:
        if Path(file_path).exists():
            if not os.access(file_path, os.X_OK):
                print(f"✗ File {file_path} is not executable")
                return False
    
    print("✓ All scripts have correct permissions")
    return True

def run_all_tests():
    """Run all tests."""
    print("Running basic structure tests...\n")
    
    try:
        if not test_directory_structure():
            return False
        if not test_required_files():
            return False
        if not test_file_permissions():
            return False
        
        print("\n✅ All basic structure tests passed!")
        print("\nProject structure is ready!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run data processing: python3 data/make_dataset.py")
        print("3. Start implementing model components!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 