import os
import numpy as np
import pandas as pd
import sys

def validate_training_input(input_dir):
    """
    Validates that input directory has ALL required files for training
    Returns: (is_valid, missing_files, warnings)
    """
    missing = []
    warnings = []
    
    if not os.path.exists(input_dir):
        return False, [f"Input directory not found: {input_dir}"], []

    # Check for 3DES files
    # We look for .npz files that are NOT RSA specific (usually traces_data.npz or similar)
    # Heuristic: if filename contains 'rsa', skip it for 3DES check
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz') and 'rsa' not in f.lower()]
    
    if not npz_files:
        missing.append("3DES .npz files (e.g. traces_data.npz)")
    else:
        # Validate 3DES .npz structure
        for npz_file in npz_files:
            try:
                data = np.load(os.path.join(input_dir, npz_file))
                # T_DES_KENC etc are required for TRAINING
                required_keys = ['trace_data', 'T_DES_KENC', 'T_DES_KMAC', 'T_DES_KDEK', 'ATC', 'Track2']
                for key in required_keys:
                    if key not in data.keys():
                        missing.append(f"{npz_file}: missing '{key}'")
            except Exception as e:
                missing.append(f"{npz_file}: Failed to load ({str(e)})")
    
    # Check for RSA files
    rsa_files = [f for f in os.listdir(input_dir) if f.endswith('.npz') and 'rsa' in f.lower()]
    if not rsa_files:
        warnings.append("No RSA .npz files found (RSA training will be skipped)")
    else:
        # Validate RSA .npz structure
        for rsa_file in rsa_files:
            try:
                data = np.load(os.path.join(input_dir, rsa_file))
                required_keys = ['trace_data', 'ACR_send', 'ACR_receive', 'Track2']
                for key in required_keys:
                    if key not in data.keys():
                        missing.append(f"{rsa_file}: missing '{key}'")
            except Exception as e:
                missing.append(f"{rsa_file}: Failed to load ({str(e)})")

    # Strict pure-ML pipeline does not rely on external spreadsheets (no KALKi Excel dependency).
    
    is_valid = len(missing) == 0
    return is_valid, missing, warnings

def validate_testing_input(input_dir):
    """
    Validates that input directory has ALL required files for testing (Attack Mode)
    Returns: (is_valid, missing_files, warnings)
    """
    missing = []
    warnings = []
    
    if not os.path.exists(input_dir):
        return False, [f"Input directory not found: {input_dir}"], []
    
    # Check for .npz files
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    if not npz_files:
        missing.append("No .npz files found in input directory")
    else:
        # Validate .npz structure
        for npz_file in npz_files:
            try:
                data = np.load(os.path.join(input_dir, npz_file))
                # For Attack, we ONLY need trace_data and Track2 (for ID) and ATC (for 3DES input)
                # ATC is critical for 3DES attack!
                required_keys = ['trace_data', 'Track2'] 
                for key in required_keys:
                    if key not in data.keys():
                        missing.append(f"{npz_file}: missing '{key}'")
                
                # Warning if ATC is missing (Attack might fail or require OCR/I/O)
                if 'ATC' not in data.keys():
                    warnings.append(f"{npz_file}: missing 'ATC' (Required for 3DES Attack unless derived)")
                
                # Check for I/O channel (RSA)
                if 'ACR_send' not in data.keys() or 'ACR_receive' not in data.keys():
                    warnings.append(f"{npz_file}: No I/O channel (RSA/PIN attack might be limited)")
                    
            except Exception as e:
                missing.append(f"{npz_file}: Failed to load ({str(e)})")
    
    is_valid = len(missing) == 0
    return is_valid, missing, warnings

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.validate_input <train|test> <input_dir>")
        sys.exit(1)
    
    mode = sys.argv[1]
    input_dir = sys.argv[2]
    
    print(f"Validating '{input_dir}' for mode: {mode.upper()}...")
    
    if mode == "train":
        is_valid, missing, warnings = validate_training_input(input_dir)
    elif mode == "test" or mode == "attack":
        is_valid, missing, warnings = validate_testing_input(input_dir)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    print("-" * 40)
    print(f"Validation Result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    print("-" * 40)
    
    if missing:
        print("\n❌ MISSING CRITICAL FILES/DATA:")
        for item in missing:
            print(f"  - {item}")
    
    if warnings:
        print("\n⚠️ WARNINGS (Non-Critical):")
        for item in warnings:
            print(f"  - {item}")
            
    if is_valid:
        print("\nSuccess: Input directory is ready for processing.")
        sys.exit(0)
    else:
        print("\nFailure: Please fix missing items before proceeding.")
        sys.exit(1)
