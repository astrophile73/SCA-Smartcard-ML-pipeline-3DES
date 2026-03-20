#!/usr/bin/env python
"""
Verify that training labels match inference challenge computation.
Check if label_type is consistent between training and attack.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"I:\freelance\SCA Smartcard ML Pipeline-3des\pipeline-code")

from src.crypto import des_sbox_output, _SBOX

processed_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\3des"

print("="*80)
print("VERIFYING TRAINING LABELS VS INFERENCE LOGIC")
print("="*80)

# Load metadata
meta_path = os.path.join(processed_dir, "Y_meta.csv")
meta_df = pd.read_csv(meta_path)

print(f"\nMetadata shape: {meta_df.shape}")
print(f"Columns: {list(meta_df.columns)}")

# Check ground truth key format
if 'T_DES_KENC' in meta_df.columns:
    sample_kenc = meta_df.iloc[0]['T_DES_KENC']
    print(f"\nGround truth KENC format: {sample_kenc}")
    print(f"  Type: {type(sample_kenc)}")
    print(f"  Length: {len(str(sample_kenc))}")
    
    try:
        int(str(sample_kenc), 16)
        print(f"  ✓ Valid hex key (16 bytes = 128 bits)")
    except:
        print(f"  ✗ NOT a valid hex key")
        
# Check if training labels exist
labels_path = os.path.join(processed_dir, "Y_labels_stage1_sbox1.npy")
if os.path.exists(labels_path):
    y_labels = np.load(labels_path)
    print(f"\nTraining labels found for S-box 1:")
    print(f"  Shape: {y_labels.shape}")
    print(f"  Dtype: {y_labels.dtype}")
    print(f"  Unique values: {np.unique(y_labels)[:10]}...")  # First 10 unique
    print(f"  Min: {y_labels.min()}, Max: {y_labels.max()}")
    
    # Check if labels are 4-bit (sbox_output) or 6-bit (sbox_input)
    max_val = y_labels.max()
    if max_val <= 15:
        print(f"  → Labels are 4-bit (sbox_output: 0-15)")
    elif max_val <= 63:
        print(f"  → Labels are 6-bit (sbox_input: 0-63)")
    else:
        print(f"  → Labels appear invalid (max={max_val})")
        
    # Show sample
    print(f"\n  Sample labels (first 20): {y_labels[:20]}")
else:
    print(f"\n✗ No training labels found")

# Check preprocessing config
config_path = os.path.join(processed_dir, "preprocessing_config.json")
if os.path.exists(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\nPreprocessing config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
else:
    print(f"\n✗ No preprocessing config found")

# Check if features and labels align
features_path = os.path.join(processed_dir, "X_features.npy")
if os.path.exists(features_path) and os.path.exists(labels_path):
    X = np.load(features_path)
    Y = np.load(labels_path)
    print(f"\nFeature-Label Alignment:")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {Y.shape}")
    if X.shape[0] == Y.shape[0]:
        print(f"  ✓ Same number of samples ({X.shape[0]})")
    else:
        print(f"  ✗ MISMATCH: {X.shape[0]} features vs {Y.shape[0]} labels")

# Now check what inference expects
print(f"\n" + "="*80)
print("INFERENCE CHALLENGE LOGIC")
print("="*80)

# Sample ATC value
if 'ATC' in meta_df.columns:
    sample_atc = meta_df.iloc[0]['ATC']
    print(f"\nSample ATC value: {sample_atc} (type: {type(sample_atc)})")
    
    # Compute challenge as inference does
    try:
        atc_int = int(sample_atc) if isinstance(sample_atc, str) else sample_atc
        # DES uses 32-bit counter
        # ER0[DES R0 input] = counter XOR constant (for S-box 1, bits 42-47)
        shift = 42  # S-box 1 uses bits 42-47 of ER0
        er0_chunk = (atc_int >> shift) & 0x3F
        print(f"  ATC as int: {atc_int}")
        print(f"  ER0 chunk (bits 42-47): 0x{er0_chunk:02X} ({er0_chunk})")
        
        # Now test S-box forward
        sample_key = meta_df.iloc[0]['T_DES_KMAC'] if 'T_DES_KMAC' in meta_df.columns else ''
        if sample_key:
            sample_key_int = int(str(sample_key), 16)
            # Extract first 6 bits of key (S-box 1 input chunk)
            k1_chunk = (sample_key_int >> (48 - 6)) & 0x3F  # First 6 bits
            print(f"  Key first 6 bits: {k1_chunk}")
            sbox_in = er0_chunk ^ k1_chunk
            print(f"  S-box 1 input: {sbox_in}")
            # Get S-box output
            b1 = (sbox_in >> 5) & 1
            b6 = sbox_in & 1
            row = (b1 << 1) | b6
            col = (sbox_in >> 1) & 0xF
            sbox_out = _SBOX[0][int(row * 16 + col)]  # S-box 1 output
            print(f"  S-box 1 output: {sbox_out}")
            print(f"  → Expected label (sbox_output mode): {sbox_out}")
    except Exception as e:
        print(f"  Error computing: {e}")

print("\n" + "="*80)
print("HYPOTHESIS")
print("="*80)
print("""
Possible reasons for model failure:

1. TRAINING LABEL MISMATCH:
   - Training labels generated with wrong ATC mapping
   - OR training labels use different key encoding
   - OR labels are sbox_input but attack expects sbox_output

2. FEATURE EXTRACTION MISMATCH:
   - Features extracted with different POIs during training vs attack
   - Normalization applied differently (per-sbox vs global)
   - S-box specific features not matching

3. MODEL ARCHITECTURE MISMATCH:
   - Models expect different input dimensions
   - Models expect different normalization scheme

ACTION: Check what was actually used to TRAIN the models
""")
