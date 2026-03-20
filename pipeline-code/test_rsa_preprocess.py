#!/usr/bin/env python
import sys, os
sys.path.insert(0, '.')
from src.pipeline_rsa import preprocess_rsa

input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
processed_rsa = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\rsa_test"
opt_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Optimization"

os.makedirs(processed_rsa, exist_ok=True)
os.makedirs(opt_dir, exist_ok=True)

print("Running RSA preprocessing test...")
print(f"  Input: {input_dir}")
print(f"  Output: {processed_rsa}")

try:
    feat_path, meta_path = preprocess_rsa(
        input_dir,
        processed_rsa,
        opt_dir,
        "traces_data_*.npz",
        "universal",
        use_existing_pois=False,
        include_secrets=True,
        enable_external_labels=False,
        label_map_xlsx=None,
        strict_label_mode=False,
        force_variance_poi=False,
        label_type="sbox_output",
        force_regenerate=False,
    )
    print(f"\n✓ SUCCESS")
    print(f"  Features: {feat_path}")
    print(f"  Metadata: {meta_path}")
    
    import pandas as pd
    if meta_path and os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        print(f"  Metadata rows: {len(df)}")
        print(f"  Columns: {list(df.columns)[:5]}")
except Exception as e:
    import traceback
    print(f"\n✗ ERROR: {e}")
    traceback.print_exc()
