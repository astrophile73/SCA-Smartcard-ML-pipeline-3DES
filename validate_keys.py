#!/usr/bin/env python3
"""
Validate extracted keys and show ground truth vs. loaded values.
"""

import sys
import os
import numpy as np
import pandas as pd

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

sys.path.insert(0, "pipeline-code")
from src.ingest import TraceDataset
from src.gen_labels import compute_labels

INPUT_DIR = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1"
PROCESSED_DIR = "3des-pipeline/Processed/3des"

print("=" * 80)
print("KEY EXTRACTION VALIDATION")
print("=" * 80)

# Load metadata using the fixed TraceDataset
try:
    print("\n[1] Loading metadata with TraceDataset...")
    ds = TraceDataset(INPUT_DIR, file_pattern="*trace*.*", card_type="universal")
    meta = ds._load_metadata()
    
    print(f"OK Loaded {len(meta)} traces")
    print(f"  Columns: {list(meta.columns)}")
    
    # Check key columns
    print(f"\n[2] Key extraction results:")
    print(f"  Total traces: {len(meta)}")
    
    n_with_kenc = (meta["T_DES_KENC"].astype(str).str.strip().astype(str) != "").sum()
    n_with_kmac = (meta["T_DES_KMAC"].astype(str).str.strip().astype(str) != "").sum()
    n_with_kdek = (meta["T_DES_KDEK"].astype(str).str.strip().astype(str) != "").sum()
    
    print(f"  Traces with T_DES_KENC: {n_with_kenc}")
    print(f"  Traces with T_DES_KMAC: {n_with_kmac}")
    print(f"  Traces with T_DES_KDEK: {n_with_kdek}")
    
    # Sample the keys
    print(f"\n[3] Sample extracted keys (first 5 traces):")
    for idx in range(min(5, len(meta))):
        kenc = meta.iloc[idx]["T_DES_KENC"]
        kmac = meta.iloc[idx]["T_DES_KMAC"]
        kdek = meta.iloc[idx]["T_DES_KDEK"]
        t2 = meta.iloc[idx]["Track2"]
        
        # Check if valid
        kenc_check = 'OK' if len(str(kenc).strip()) == 32 else 'FAIL'
        kmac_check = 'OK' if len(str(kmac).strip()) == 32 else 'FAIL'
        kdek_check = 'OK' if len(str(kdek).strip()) == 32 else 'FAIL'
        
        print(f"\n  Row {idx}:")
        print(f"    Track2: {str(t2)[:20]}...")
        print(f"    KENC ({len(str(kenc))} chars): {kenc_check} {str(kenc)[:32]}")
        print(f"    KMAC ({len(str(kmac))} chars): {kmac_check} {str(kmac)[:32]}")
        print(f"    KDEK ({len(str(kdek))} chars): {kdek_check} {str(kdek)[:32]}")
    
    # Test label computation
    print(f"\n[4] Testing label computation:")
    sample_meta = meta.iloc[:3].copy()
    
    for idx in range(len(sample_meta)):
        kenc = sample_meta.iloc[idx]["T_DES_KENC"]
        
        # Try computing labels
        try:
            labels = compute_labels(sample_meta.iloc[idx:idx+1], sbox_idx=0, key_col="T_DES_KENC", stage=1)
            is_valid = labels[0] >= 0 and labels[0] < 16
            status = "[OK] Valid" if is_valid else "[FAIL] Invalid"
            print(f"  Row {idx}: {status} - Label={labels[0]}")
        except Exception as e:
            print(f"  Row {idx}: [ERR] {e}")
    
    # Check if processed directory has features
    print(f"\n[5] Checking processed features:")
    if os.path.exists(PROCESSED_DIR):
        files = os.listdir(PROCESSED_DIR)
        print(f"  Files in {PROCESSED_DIR}:")
        for f in sorted(files):
            fpath = os.path.join(PROCESSED_DIR, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                print(f"    {f} ({size} bytes)")
    else:
        print(f"  [MISSING] {PROCESSED_DIR}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
