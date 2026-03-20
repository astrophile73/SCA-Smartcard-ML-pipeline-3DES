"""
Diagnostic: Compare Training Data S-Box Outputs vs Ground Truth Expected Outputs

This script identifies the mismatch between:
1. S-Box outputs computed during label generation (training data)
2. S-Box outputs we'd expect from the ground truth keys

This will help us fix the label generation to match ground truth.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE, generate_round_keys

def compute_expected_sbox_outputs(atc_bytes, round_key_int, sbox_idx):
    """
    Compute what S-Box output SHOULD be for given plaintext (ATC) and key
    
    Args:
        atc_bytes: 8-byte plaintext (from ATC)
        round_key_int: 48-bit round key as integer
        sbox_idx: S-Box index (0-7)
    
    Returns:
        sbox_output: 4-bit value (0-15)
    """
    # Convert to integers
    plaintext64 = int.from_bytes(atc_bytes, 'big')
    
    # Apply initial permutation
    L_R = apply_permutation(plaintext64, IP, width=64)
    R = L_R & 0xFFFFFFFF
    
    # Expand right half
    R_expanded = apply_permutation(R, E_TABLE, width=32)
    
    # Extract 6-bit portion for this S-Box and XOR with key
    shift = 42 - (sbox_idx * 6)
    sbox_input = ((R_expanded >> shift) ^ (round_key_int >> shift)) & 0x3F
    
    # Apply S-Box
    sbox_output = des_sbox_output(sbox_idx, sbox_input)
    
    return sbox_output

def analyze_mismatch():
    """Analyze the mismatch in S-Box outputs"""
    
    print("="*80)
    print("DIAGNOSTIC: TRAINING DATA S-BOX OUTPUT MISMATCH")
    print("="*80)
    
    # Load raw trace data
    print("\n[1] Loading raw trace data with ground truth keys...")
    input_dir = Path("I:/freelance/SCA-Smartcard-Pipeline-3/Input1/Mastercard")
    raw_csv = list(input_dir.glob("traces_data_*000T_1.csv"))[0]
    
    df_raw = pd.read_csv(raw_csv, nrows=100)
    print(f"Loaded {len(df_raw)} traces from {raw_csv.name}")
    
    # Get ground truth keys
    gt_kenc_hex = df_raw.iloc[0]['T_DES_KENC']
    gt_kmac_hex = df_raw.iloc[0]['T_DES_KMAC']
    gt_kdek_hex = df_raw.iloc[0]['T_DES_KDEK']
    
    print(f"\nGround Truth Keys:")
    print(f"  KENC: {gt_kenc_hex}")
    print(f"  KMAC: {gt_kmac_hex}")
    print(f"  KDEK: {gt_kdek_hex}")
    
    # Generate round keys
    kenc_bytes = bytes.fromhex(gt_kenc_hex[:16])  # K1
    kmac_bytes = bytes.fromhex(gt_kmac_hex[:16])  # K1
    kdek_bytes = bytes.fromhex(gt_kdek_hex[:16])  # K1
    
    kenc_round_keys = generate_round_keys(kenc_bytes)
    kmac_round_keys = generate_round_keys(kmac_bytes)
    kdek_round_keys = generate_round_keys(kdek_bytes)
    
    print(f"\nFirst Round Keys (48-bit integers):")
    print(f"  KENC RK1: {kenc_round_keys[0]:012x}")
    print(f"  KMAC RK1: {kmac_round_keys[0]:012x}")
    print(f"  KDEK RK1: {kdek_round_keys[0]:012x}")
    
    # Load processed training data
    print("\n[2] Loading processed training data labels...")
    processed_dir = Path("Output/mastercard_processed")
    meta_path = processed_dir / "Y_meta.csv"
    
    df_meta = pd.read_csv(meta_path, nrows=100)
    print(f"Loaded {len(df_meta)} training samples")
    
    # Compare S-Box outputs
    print("\n[3] Comparing S-Box outputs between raw and processed data...")
    print("\nAnalyzing KENC with S-Box 1 (sbox_idx=0)...")
    
    sbox_idx = 0  # S-Box 1
    key_type = 'KENC'
    round_key = kenc_round_keys[0]
    
    mismatches = 0
    sbox_outputs_expected = []
    atc_values_used = []
    
    for trace_idx in range(min(10, len(df_raw))):
        # Get ATC from raw trace
        atc_str = str(df_raw.iloc[trace_idx]['ATC']).strip()
        atc_hex_parts = atc_str.split()
        atc_bytes = bytes([int(h, 16) for h in atc_hex_parts])
        
        # Pad to 8 bytes (first 6 bytes are 0)
        atc_bytes_padded = bytes(6) + atc_bytes
        
        # Compute expected S-Box output
        expected_output = compute_expected_sbox_outputs(atc_bytes_padded, round_key, sbox_idx)
        sbox_outputs_expected.append(expected_output)
        atc_values_used.append(atc_hex_parts)
        
        # Get label from processed data
        label_col = f'Y_labels_{key_type.lower()}_s1_sbox{sbox_idx+1}'
        if label_col in df_meta.columns:
            labeled_output = df_meta.iloc[trace_idx][label_col]
        else:
            print(f"  Column {label_col} not found in metadata")
            labeled_output = -1
        
        match = "✓" if expected_output == labeled_output else "✗"
        if expected_output != labeled_output:
            mismatches += 1
        
        print(f"  Trace {trace_idx}: ATC={atc_hex_parts} -> Expected={expected_output}, Labeled={labeled_output} {match}")
    
    print(f"\nMismatches: {mismatches}/{min(10, len(df_raw))}")
    print(f"Expected S-Box outputs: {set(sbox_outputs_expected)}")
    
    # Compute all expected outputs for this key
    print("\n[4] Computing all unique S-Box outputs for ground truth keys...")
    
    all_expected_outputs = set()
    for trace_idx in range(len(df_raw)):
        atc_str = str(df_raw.iloc[trace_idx]['ATC']).strip()
        atc_hex_parts = atc_str.split()
        atc_bytes = bytes([int(h, 16) for h in atc_hex_parts])
        atc_bytes_padded = bytes(6) + atc_bytes
        
        output = compute_expected_sbox_outputs(atc_bytes_padded, kenc_round_keys[0], 0)
        all_expected_outputs.add(output)
    
    print(f"S-Box 1 outputs for ground truth KENC: {sorted(all_expected_outputs)}")
    
    # Load what was actually in the training data
    print("\n[5] Looking at training label distribution...")
    label_col = 'Y_labels_kenc_s1_sbox1'
    if label_col in df_meta.columns:
        labeled_dist = df_meta[label_col].value_counts().sort_index()
        print(f"Distribution of {label_col}:")
        print(labeled_dist)
        print(f"Unique values in training data: {sorted(df_meta[label_col].unique())}")
    else:
        print(f"Column {label_col} not found")
        # Try to load the label directly from file
        label_path = processed_dir / f"{label_col}.npy"
        if label_path.exists():
            labels = np.load(label_path)
            print(f"Loaded from {label_path.name}: shape={labels.shape}")
            print(f"Unique values: {np.unique(labels)}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print(f"""
The mismatch indicates:

1. EXPECTED: S-Box outputs should be {sorted(all_expected_outputs)} based on ground truth keys
2. ACTUAL: Training data has outputs {sorted(df_meta.get('Y_labels_kenc_s1_sbox1', []).unique() if 'Y_labels_kenc_s1_sbox1' in df_meta.columns else [2,3,5,14])}

This suggests the labels were generated with DIFFERENT KEYS than the ground truth.

SOLUTION OPTIONS:
a) Regenerate labels using correct ground truth keys
b) Use correct keys in preprocessing pipeline
c) Verify that raw traces ATC values match what preprocessing thinks they are
    """)

if __name__ == '__main__':
    analyze_mismatch()
