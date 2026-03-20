"""
Debug script to check what models are predicting vs ground truth.
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "I:\\freelance\\SCA Smartcard ML Pipeline-3des\\pipeline-code")

from src.inference_3des import recover_3des_keys
from src.crypto import _SBOX

# Configuration
processed_dir = "I:\\freelance\\SCA Smartcard ML Pipeline-3des\\3des-pipeline\\Processed\\3des"
model_dir = "I:\\freelance\\SCA Smartcard ML Pipeline-3des\\3des-pipeline\\models"

# Load ground truth
meta_path = os.path.join(processed_dir, "Y_meta.csv")
meta = pd.read_csv(meta_path)

print("="*80)
print("GROUND TRUTH KEYS")
print("="*80)
print(f"Total traces: {len(meta)}")
print(f"\nUnique KENC values: {meta['T_DES_KENC'].nunique()}")
print(f"Unique KMAC values: {meta['T_DES_KMAC'].nunique()}")
print(f"Unique KDEK values: {meta['T_DES_KDEK'].nunique()}")

print(f"\nGround truth keys (first 10 different rows):")
unique_keys = meta[['T_DES_KENC', 'T_DES_KMAC', 'T_DES_KDEK']].drop_duplicates()
print(unique_keys.head(10))

# Get predicted keys
print("\n" + "="*80)
print("PREDICTED KEYS")
print("="*80)
pred_dict = recover_3des_keys(processed_dir, model_dir, card_type="universal", n_attack=100)
print(f"Predicted keys: {pred_dict}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
if '3DES_KENC' in pred_dict and len(unique_keys) > 0:
    pred_kenc = pred_dict['3DES_KENC']
    true_kenc = unique_keys.iloc[0]['T_DES_KENC']
    print(f"Predicted KENC:  {pred_kenc}")
    print(f"Ground truth KENC: {true_kenc}")
    print(f"Match: {pred_kenc == true_kenc}")
    
    # Check byte by byte
    print("\nByte-by-byte comparison:")
    for i in range(0, len(pred_kenc), 2):
        pred_byte = pred_kenc[i:i+2]
        true_byte = true_kenc[i:i+2]
        match = "✓" if pred_byte == true_byte else "✗"
        print(f"  Byte {i//2:2d}: {match} Pred={pred_byte} True={true_byte}")

# Check Label generation
print("\n" + "="*80)
print("LABEL STATISTICS")
print("="*80)
label_file = os.path.join(processed_dir, "Y_labels_kenc_s1_sbox1.npy")
if os.path.exists(label_file):
    labels = np.load(label_file)
    print(f"S1 KENC Sbox1 labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Label distribution:")
    for val in np.unique(labels):
        count = np.sum(labels == val)
        print(f"  Class {val:2d}: {count:6d} ({100*count/len(labels):5.1f}%)")
