#!/usr/bin/env python
"""
Debug the 3DES inference process to see what's going wrong.
Check intermediate predictions before voting/reconstruction.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import traceback

sys.path.insert(0, r"I:\freelance\SCA Smartcard ML Pipeline-3des\pipeline-code")

from src.inference_3des import recover_3des_master_key

# Test data path
data_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Input"
processed_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\3des"
models_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\models\3des"

print("="*80)
print("DEBUG: 3DES Model Inference")
print("="*80)

try:
    # Check if features exist
    x_features_path = os.path.join(processed_dir, "X_features.npy")
    if os.path.exists(x_features_path):
        X_features = np.load(x_features_path)
        print(f"\n✓ Features loaded: shape={X_features.shape}")
        print(f"  Feature dtype: {X_features.dtype}")
        print(f"  Feature stats: min={X_features.min():.4f}, max={X_features.max():.4f}, mean={X_features.mean():.4f}")
        print(f"  Feature sample (row 0, first 5): {X_features[0, :5]}")
    else:
        print(f"✗ Features not found: {x_features_path}")
        
    # Load metadata to check ground truth
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    meta_df = pd.read_csv(meta_path)
    print(f"\n✓ Metadata loaded: {len(meta_df)} rows")
    
    # Check available models
    print(f"\n✓ Checking models in: {models_dir}")
    for key_type in ['kenc', 'kmac', 'kdek']:
        key_dir = os.path.join(models_dir, key_type)
        if os.path.exists(key_dir):
            print(f"\n  {key_type.upper()}:")
            for sbox in range(1, 9):
                sbox_models = []
                for model_idx in range(3):
                    model_path = os.path.join(key_dir, f"s{sbox}_model{model_idx}.pth")
                    if os.path.exists(model_path):
                        sbox_models.append(model_path)
                print(f"    S{sbox}: {len(sbox_models)} models found")
        else:
            print(f"  {key_type.upper()}: Directory not found")
    
    # Try calling recover_3des_master_key
    print(f"\n" + "="*80)
    print("Testing recover_3des_master_key()...")
    print("="*80)
    
    # We need to check what recover_3des_master_key returns
    # This requires running the actual inference
    
    recovered = recover_3des_master_key(
        feature_dir=processed_dir,
        models_dir=models_dir,
        num_traces=1000,  # First 1000 traces
        card_type="MasterCard"  # Assuming this was the training card type
    )
    
    print(f"\n✓ Recovered keys:")
    for key_type in ['3DES_KENC', '3DES_KMAC', '3DES_KDEK']:
        if key_type in recovered:
            vals = recovered[key_type]
            if isinstance(vals, list) and len(vals) > 0:
                print(f"  {key_type}: {vals[0]}")
            else:
                print(f"  {key_type}: {vals}")
    
    # Compare with ground truth
    print(f"\n✓ Ground Truth (from metadata):")
    gt_kenc = meta_df.iloc[0]['T_DES_KENC']
    gt_kmac = meta_df.iloc[0]['T_DES_KMAC']
    gt_kdek = meta_df.iloc[0]['T_DES_KDEK']
    print(f"  KENC: {gt_kenc}")
    print(f"  KMAC: {gt_kmac}")
    print(f"  KDEK: {gt_kdek}")
    
    # Check model predictions in detail (add instrumentation)
    print(f"\n" + "="*80)
    print("Checking model prediction outputs...")
    print("="*80)
    print("\nThis requires adding debug logging to inference_3des.py")
    print("The issue likely stems from:")
    print("1. Wrong models loaded (not trained on matching data)")
    print("2. Feature mismatch (different normalization)")
    print("3. Inference returning wrong prediction format")
    print("4. Reconstruction logic error (all keys identical = reconstruction bug)")
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()
