#!/usr/bin/env python
"""
Generate and save 3DES training labels that training_ensemble.py expects.
This was missing in the pipeline, causing training to skip all S-boxes.
"""
import os
import sys
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, r"I:\freelance\SCA Smartcard ML Pipeline-3des\pipeline-code")

from src.crypto import des_sbox_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processed_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\3des"

print("="*80)
print("GENERATING MISSING 3DES TRAINING LABELS")
print("="*80)

try:
    # Load metadata to get ATC values and ground truth keys
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    meta_df = pd.read_csv(meta_path)
    
    print(f"\nLoaded metadata: {len(meta_df)} traces")
    
    # Load features
    features_path = os.path.join(processed_dir, "X_features.npy")
    X_features = np.load(features_path)
    print(f"Features shape: {X_features.shape}")
    
    # Parse ATC and keys
    atc_values = []
    for idx, row in meta_df.iterrows():
        atc_str = str(row['ATC']).strip()
        # Parse hex string "7A CD" -> 0x7ACD
        atc_hex = atc_str.replace(' ', '')
        if len(atc_hex) > 0:
            atc_values.append(int(atc_hex, 16))
        else:
            atc_values.append(0)
    
    atc_values = np.array(atc_values, dtype=np.uint32)
    print(f"Parsed ATC values: min={atc_values.min()}, max={atc_values.max()}")
    
    # Generate labels for each key type and S-box
    for key_type in ['kenc', 'kmac', 'kdek']:
        print(f"\n--- Generating labels for {key_type.upper()} ---")
        
        # Get ground truth key column
        key_col = f'T_DES_{key_type.upper()}'  # T_DES_KENC, T_DES_KMAC, T_DES_KDEK
        if key_col not in meta_df.columns:
            print(f"  ERROR - Column {key_col} not found")
            continue
            
        ground_truth_keys = []
        for idx, row in meta_df.iterrows():
            key_hex = str(row[key_col]).strip()
            if key_hex and key_hex != '':
                # 3DES key is 128 bits, split into 64-bit chunks for processing
                # For S-box labeling, we only need the first 48 bits (for RK1)
                try:
                    # Parse as int, but handle 128-bit properly
                    key_int = int(key_hex, 16)
                    # Extract first 64 bits (most significant)
                    key_first_64 = (key_int >> 64) & 0xFFFFFFFFFFFFFFFF
                    ground_truth_keys.append(key_first_64)
                except:
                    ground_truth_keys.append(0)
            else:
                ground_truth_keys.append(0)
        
        ground_truth_keys = np.array(ground_truth_keys, dtype=np.uint64)
        
        # Generate Stage 1 labels
        print(f"  Generating Stage 1 labels...")
        for sbox_idx in range(1, 9):
            print(f"    S-box {sbox_idx}...", end=" ")
            
            # Compute Round Key 1 (first 6 bits for each S-box)
            shift = 42 - ((sbox_idx - 1) * 6)
            
            # For each trace, compute the S-box output label
            labels = np.full(len(meta_df), -1, dtype=np.int64)
            
            for trace_idx, (atc_val, key_val) in enumerate(zip(atc_values, ground_truth_keys)):
                # Expanded Round 0 (simplified: use ATC as challenge)
                # Real DES: ER0 = permuted(R0), here using counter
                er0_val = int(atc_val)  # Convert numpy value to Python int
                
                # Extract 6-bit chunk for this S-box from counter
                er0_chunk = (er0_val >> shift) & 0x3F
                
                # Extract 6-bit key chunk for this S-box (RK1)
                k1_chunk = (int(key_val) >> shift) & 0x3F  # Convert numpy value to Python int
                
                # S-box input = ER0 XOR RK1
                sbox_in = er0_chunk ^ k1_chunk
                
                # S-box output
                b1 = (sbox_in >> 5) & 1
                b6 = sbox_in & 1
                row = (b1 << 1) | b6
                col = (sbox_in >> 1) & 0xF
                sbox_out = des_sbox_output(sbox_idx - 1, sbox_in)  # S-box is 0-indexed in array
                
                labels[trace_idx] = sbox_out
            
            # Save labels
            output_path = os.path.join(processed_dir, f"Y_labels_{key_type}_s1_sbox{sbox_idx}.npy")
            np.save(output_path, labels)
            
            unique_labels = len(np.unique(labels))
            print(f"OK - {output_path}")
            print(f"      Unique labels: {unique_labels}, Range: [{labels.min()}, {labels.max()}]")
    
    print(f"\n" + "="*80)
    print("LABELS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nNow run training again with: --mode train")
    print("The training will now find the labels and actually train the models.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
