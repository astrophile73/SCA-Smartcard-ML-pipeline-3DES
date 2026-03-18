#!/usr/bin/env python3
"""
Identify which trace files have valid keys.
"""

import sys
import os
import numpy as np
import pandas as pd
import glob

sys.path.insert(0, "pipeline-code")
from src.ingest import TraceDataset

INPUT_DIR = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1"

print("=" * 80)
print("TRACE FILES KEY ANALYSIS")
print("=" * 80)

# Find all trace files recursively
csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "**/*.csv"), recursive=True))
npz_files = sorted(glob.glob(os.path.join(INPUT_DIR, "**/*.npz"), recursive=True))

print(f"\nFound {len(csv_files)} CSV files, {len(npz_files)} NPZ files\n")

# Analyze each file
file_stats = []

for fpath in csv_files[:15]:  # First 15 CSV files
    fname = os.path.basename(fpath)
    try:
        df = pd.read_csv(fpath, nrows=1)
        n_rows = len(pd.read_csv(fpath))
        
        has_kenc = "T_DES_KENC" in df.columns
        has_kmac = "T_DES_KMAC" in df.columns  
        has_kdek = "T_DES_KDEK" in df.columns
        
        # Check if first row has valid keys
        if has_kenc and has_kmac and has_kdek:
            kenc = str(df.iloc[0]["T_DES_KENC"]).strip()
            kmac = str(df.iloc[0]["T_DES_KMAC"]).strip()
            kdek = str(df.iloc[0]["T_DES_KDEK"]).strip()
            
            kenc_valid = len(kenc) == 32
            kmac_valid = len(kmac) == 32
            kdek_valid = len(kdek) == 32
            
            key_status = "OK" if (kenc_valid and kmac_valid and kdek_valid) else "EMPTY/INVALID"
        else:
            key_status = "NO_COLS"
        
        file_stats.append({
            'file': fname,
            'rows': n_rows,
            'has_cols': has_kenc and has_kmac and has_kdek,
            'key_status': key_status
        })
        
    except Exception as e:
        file_stats.append({
            'file': fname,
            'rows': '?',
            'has_cols': False,
            'key_status': f'ERROR: {e}'
        })

# Print summary
print("[CSV FILES WITH KEYS]")
for stat in file_stats:
    print(f"  {stat['file']:40} | Rows: {stat['rows']:6} | Cols: {stat['has_cols']} | Keys: {stat['key_status']}")

print("\n" + "=" * 80)
