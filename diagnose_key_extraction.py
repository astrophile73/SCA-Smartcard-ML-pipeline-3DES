#!/usr/bin/env python3
"""
Diagnose key extraction accuracy from trace files.
Check if ground truth keys are being loaded and used correctly.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, "pipeline-code")
from src.ingest import TraceDataset
from src.gen_labels import compute_labels

INPUT_DIR = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1"

print("=" * 80)
print("KEY EXTRACTION ACCURACY DIAGNOSIS")
print("=" * 80)

# Find trace files
import glob
csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
npz_files = glob.glob(os.path.join(INPUT_DIR, "*.npz"))

print(f"\nFound {len(csv_files)} CSV files, {len(npz_files)} NPZ files")

# Check first CSV file
if csv_files:
    csv_file = sorted(csv_files)[0]
    print(f"\n{'='*80}")
    print(f"CSV File: {os.path.basename(csv_file)}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(csv_file, nrows=5)
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst row:")
        for col in df.columns:
            val = df.iloc[0][col]
            if isinstance(val, str) and len(str(val)) > 100:
                print(f"  {col}: {str(val)[:100]}... (length: {len(str(val))})")
            else:
                print(f"  {col}: {val}")
        
        # Check key columns
        key_cols = ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK"]
        print(f"\n3DES Keys from CSV:")
        for col in key_cols:
            if col in df.columns:
                val = df.iloc[0][col]
                print(f"  {col}: {val}")
                # Check if it's a valid hex key
                try:
                    hex_str = str(val).strip().replace(" ", "").upper()
                    if len(hex_str) == 32:
                        print(f"    ✓ Valid 32-char hex")
                    else:
                        print(f"    ✗ Invalid length: {len(hex_str)} (expected 32)")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
        
        # Check trace_data column
        if "trace_data" in df.columns:
            trace_val = df.iloc[0]["trace_data"]
            print(f"\ntrace_data column:")
            print(f"  Type: {type(trace_val)}")
            trace_str = str(trace_val)[:100]
            print(f"  First 100 chars: {trace_str}")
            
            # Try to parse as floats
            try:
                if isinstance(trace_val, str):
                    floats = [float(x) for x in trace_val.split(",")]
                    print(f"  ✓ Parsed as {len(floats)} floats")
                    print(f"    Range: [{min(floats):.2f}, {max(floats):.2f}]")
                else:
                    print(f"  Already numeric (not string)")
            except Exception as e:
                print(f"  ✗ Cannot parse: {e}")
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        import traceback
        traceback.print_exc()

# Check NPZ file
if npz_files:
    npz_file = sorted(npz_files)[0]
    print(f"\n{'='*80}")
    print(f"NPZ File: {os.path.basename(npz_file)}")
    print(f"{'='*80}")
    
    try:
        data = np.load(npz_file, allow_pickle=True)
        print(f"Keys: {list(data.files)}")
        
        # Check key arrays
        key_cols = ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK"]
        print(f"\n3DES Keys from NPZ:")
        for col in key_cols:
            if col in data:
                val = data[col]
                print(f"  {col}:")
                print(f"    Type: {type(val)}, Shape: {val.shape}, DType: {val.dtype}")
                try:
                    if val.ndim == 0:
                        first_val = str(val.item())
                    else:
                        first_val = str(val[0])
                    print(f"    First value: {first_val}")
                    
                    # Check if valid hex
                    hex_str = first_val.strip().replace(" ", "").upper()
                    if len(hex_str) == 32:
                        print(f"    ✓ Valid 32-char hex")
                    else:
                        print(f"    ✗ Invalid length: {len(hex_str)} (expected 32)")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
        
        # Check trace_data
        if "trace_data" in data:
            trace_val = data["trace_data"]
            print(f"\ntrace_data array:")
            print(f"  Shape: {trace_val.shape}, DType: {trace_val.dtype}")
            if trace_val.ndim > 1:
                print(f"  ✓ 2D array suitable for traces ({trace_val.shape[0]} traces, {trace_val.shape[1]} samples)")
            elif trace_val.ndim == 1:
                print(f"  ? 1D array (might be single trace or metadata)")
                print(f"  First 10 values: {trace_val[:10]}")
            else:
                print(f"  ✗ Scalar array - unexpected format")
                
    except Exception as e:
        print(f"Error reading NPZ: {e}")
        import traceback
        traceback.print_exc()

# Try using TraceDataset to see if keys are loaded correctly
print(f"\n{'='*80}")
print("Testing TraceDataset key loading:")
print(f"{'='*80}")

try:
    ds = TraceDataset(INPUT_DIR, file_pattern="*trace*.*", card_type="universal")
    meta = ds._load_metadata()
    
    print(f"\nLoaded metadata shape: {meta.shape}")
    print(f"Columns: {list(meta.columns)}")
    
    # Check first few rows for ground truth keys
    print(f"\nGround truth keys from metadata (first 3 rows):")
    for idx in range(min(3, len(meta))):
        print(f"\n  Row {idx}:")
        for col in ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK"]:
            if col in meta.columns:
                val = meta.iloc[idx][col]
                print(f"    {col}: {val}")
    
    # Try computing labels for first trace
    print(f"\n\nTesting label computation (first trace):")
    meta_sample = meta.iloc[:1].copy()
    for sbox_idx in range(8):
        labels = compute_labels(meta_sample, sbox_idx=sbox_idx, key_col="T_DES_KENC", stage=1)
        print(f"  S-Box {sbox_idx+1}: label={labels[0]} (0-15 expected)")
        if labels[0] == -1:
            print(f"    ✗ FAILED TO COMPUTE - Missing/invalid key")
        elif not (0 <= labels[0] <= 15):
            print(f"    ✗ INVALID OUTPUT - Expected 0-15, got {labels[0]}")
        else:
            print(f"    ✓ Valid label")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("END DIAGNOSIS")
print(f"{'='*80}")
