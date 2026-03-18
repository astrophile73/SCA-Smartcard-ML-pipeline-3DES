#!/usr/bin/env python3
"""Diagnose what keys are actually in the training input traces"""
import numpy as np
import pandas as pd
import os
from glob import glob

input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1"

print("=" * 80)
print("DIAGNOSING TRAINING INPUT TRACES")
print("=" * 80)

# Find all trace files
npz_files = glob(os.path.join(input_dir, "**", "*.npz"), recursive=True)
csv_files = glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)

print(f"\nFound NPZ files: {len(npz_files)}")
print(f"Found CSV files: {len(csv_files)}")

# Check NPZ files for 3DES keys
print("\n=== NPZ Files (3DES) ===")
for npz_file in npz_files[:5]:  # Check first 5
    print(f"\nFile: {os.path.basename(npz_file)}")
    try:
        data = np.load(npz_file, allow_pickle=True)
        print(f"  Keys available: {list(data.files)}")
        
        # Check for 3DES key fields
        for key in ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK"]:
            if key in data:
                vals = data[key]
                print(f"  {key}: {vals[0] if len(vals) > 0 else 'empty'}")
        
        if "Track2" in data:
            track2 = data["Track2"]
            print(f"  Track2 (first 3): {track2[:3]}")
    except Exception as e:
        print(f"  Error: {e}")

# Check CSV files
print("\n=== CSV Files ===")
for csv_file in csv_files[:5]:  # Check first 5
    print(f"\nFile: {os.path.basename(csv_file)}")
    try:
        df = pd.read_csv(csv_file, nrows=3)
        print(f"  Columns: {list(df.columns)}")
        
        # Show first row of 3DES keys if present
        for col in ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK", "3DES_KENC", "3DES_KMAC", "3DES_KDEK"]:
            if col in df.columns:
                print(f"  {col} (first): {df[col].iloc[0]}")
        
        if "Track2" in df.columns:
            print(f"  Track2 (first): {df['Track2'].iloc[0]}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("END DIAGNOSIS")
print("=" * 80)
