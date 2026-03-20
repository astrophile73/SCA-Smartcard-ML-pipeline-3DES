#!/usr/bin/env python
"""
Diagnostic script to identify:
1. What 3DES attack returns (master key vs per-row keys)
2. Whether RSA models exist and can be loaded
3. What RSA attack returns
"""
import os
import sys
import pandas as pd

# Paths from test7_fixed
processed_3des = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\3des"
processed_rsa = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\rsa"
model_root = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\models"

print("=" * 80)
print("DIAGNOSTIC: 3DES Attack Output")
print("=" * 80)

meta_3des = os.path.join(processed_3des, "Y_meta.csv")
if os.path.exists(meta_3des):
    df = pd.read_csv(meta_3des)
    print(f"✓ 3DES metadata: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
else:
    print(f"✗ 3DES metadata missing: {meta_3des}")

# Check for 3DES models
model_3des_dir = os.path.join(model_root, "3des")
if os.path.isdir(model_3des_dir):
    print(f"✓ 3DES model directory exists: {model_3des_dir}")
    for item in os.listdir(model_3des_dir):
        item_path = os.path.join(model_3des_dir, item)
        if os.path.isdir(item_path):
            print(f"  - {item}/ ({len(os.listdir(item_path))} files)")
        else:
            print(f"  - {item}")
else:
    print(f"✗ 3DES model directory missing: {model_3des_dir}")

print("\n" + "=" * 80)
print("DIAGNOSTIC: RSA Attack Output")
print("=" * 80)

meta_rsa = os.path.join(processed_rsa, "Y_meta.csv")
if os.path.exists(meta_rsa):
    df = pd.read_csv(meta_rsa)
    print(f"✓ RSA metadata: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
else:
    print(f"✗ RSA metadata missing: {meta_rsa}")

features_rsa = os.path.join(processed_rsa, "X_features.npy")
if os.path.exists(features_rsa):
    import numpy as np
    X = np.load(features_rsa)
    print(f"✓ RSA features exist: shape {X.shape}")
else:
    print(f"✗ RSA features missing: {features_rsa}")

# Check for RSA models
model_ensemble_dir = os.path.join(model_root, "Ensemble_RSA")
if os.path.isdir(model_ensemble_dir):
    print(f"✓ Ensemble RSA models directory: {model_ensemble_dir}")
    files = os.listdir(model_ensemble_dir)
    print(f"  Files: {sorted(files)}")
else:
    print(f"✗ Ensemble RSA directory missing")

print("\n" + "=" * 80)
print("DIAGNOSTIC: Test Attack Functions")
print("=" * 80)

# Try 3DES attack
try:
    from src.pipeline_3des import attack_3des
    print("Testing 3DES attack...")
    predicted_3des, final_key = attack_3des(
        processed_3des,
        model_root,
        card_type="mastercard",
        target_key="session"
    )
    
    if predicted_3des:
        print(f"✓ 3DES attack returned predictions:")
        for key, val in predicted_3des.items():
            if isinstance(val, list):
                print(f"  {key}: list of {len(val)} items")
                if len(val) > 0:
                    # Check if all items are the same (master key repeated)
                    if all(v == val[0] for v in val):
                        print(f"    → All SAME (master key): {val[0][:32]}...")
                    else:
                        print(f"    → DIFFERENT per row:")
                        for i in range(min(3, len(val))):
                            print(f"       Row {i}: {val[i][:32]}...")
            else:
                print(f"  {key}: {val}")
    else:
        print(f"✗ 3DES attack returned None for predicted_3des, final_key={final_key}")
        
except Exception as e:
    print(f"✗ 3DES attack error: {e}")
    import traceback
    traceback.print_exc()

# Try RSA attack  
try:
    from src.pipeline_rsa import attack_rsa
    print("\nTesting RSA attack...")
    rsa_predictions, pin = attack_rsa(
        processed_rsa,
        model_root,
        meta_path=meta_rsa,
        run_pin=False
    )
    
    if rsa_predictions:
        print(f"✓ RSA attack returned predictions:")
        for key, val in rsa_predictions.items():
            if isinstance(val, list):
                print(f"  {key}: list of {len(val)} items")
                if len(val) > 0:
                    empty_count = sum(1 for v in val if not v or v == "")
                    print(f"    → {empty_count} empty, {len(val) - empty_count} populated")
                    for i in range(min(2, len(val))):
                        if val[i]:
                            print(f"       Row {i}: {val[i][:32]}...")
                        else:
                            print(f"       Row {i}: EMPTY")
    else:
        print(f"✗ RSA attack returned None")
        
except Exception as e:
    print(f"✗ RSA attack error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
