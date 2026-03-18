#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare training data: Filter traces for 3DES only (Mastercard + Greenvisa)
Creates symlinks or copies of only the needed traces for retraining
"""

import os
import sys
from pathlib import Path
import shutil
import pandas as pd

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def print_header(msg):
    print("\n" + "=" * 90)
    print(f" {msg}")
    print("=" * 90)

def identify_trace_type(filepath):
    """
    Identify if a trace file is 3DES or RSA based on content
    """
    try:
        df = pd.read_csv(filepath, nrows=1)
        cols = df.columns.tolist()
        
        # 3DES indicators: Has 3DES_* columns or specific structure
        is_3des = any('3DES' in str(c).upper() or '3des' in str(c).lower() for c in cols)
        is_rsa = any('RSA' in str(c).upper() or 'rsa' in str(c).lower() for c in cols)
        
        # If ambiguous, check column count or specific patterns
        if not is_3des and not is_rsa:
            # 3DES typically has fewer columns (trace data + metadata)
            # RSA might have more (multi-component)
            is_3des = True  # Default to 3DES if unclear
        
        return '3DES' if is_3des else 'RSA'
    except:
        return None

def collect_3des_traces():
    """
    Collect all 3DES traces from Mastercard and Greenvisa
    """
    print_header("COLLECTING 3DES TRACES")
    
    base_dir = Path("I:\\freelance\\SCA-Smartcard-Pipeline-3\\Input1")
    
    traces = {
        'mastercard': [],
        'greenvisa': []
    }
    
    # Mastercard traces
    mc_dir = base_dir / "Mastercard"
    if mc_dir.exists():
        print(f"\nScanning Mastercard directory: {mc_dir}")
        for f in mc_dir.glob("*.csv"):
            trace_type = identify_trace_type(f)
            if trace_type == '3DES':
                traces['mastercard'].append(f)
                print(f"  [OK] {f.name} (3DES)")
            else:
                print(f"  [--] {f.name} (not 3DES, skipping)")
    else:
        print(f"[-] Mastercard directory not found: {mc_dir}")
    
    # Greenvisa traces
    gv_dir = base_dir / "Visa" / "Green Visa Traces - 5000 (3DES)"
    if gv_dir.exists():
        print(f"\nScanning Greenvisa directory: {gv_dir}")
        for f in gv_dir.glob("*.csv"):
            trace_type = identify_trace_type(f)
            if trace_type == '3DES':
                traces['greenvisa'].append(f)
                print(f"  [OK] {f.name} (3DES)")
            else:
                print(f"  [--] {f.name} (not 3DES, skipping)")
    else:
        print(f"[-] Greenvisa directory not found: {gv_dir}")
    
    print(f"\nSummary:")
    print(f"  Mastercard 3DES traces: {len(traces['mastercard'])}")
    print(f"  Greenvisa 3DES traces: {len(traces['greenvisa'])}")
    print(f"  Total 3DES traces: {len(traces['mastercard']) + len(traces['greenvisa'])}")
    
    return traces

def prepare_training_input_dir(traces):
    """
    Create a training input directory with only 3DES traces
    Uses symlinks to avoid copying large files
    """
    print_header("PREPARING TRAINING INPUT DIRECTORY")
    
    training_input = Path("3des-pipeline/Input_3DES_Training")
    training_input.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating training input directory: {training_input}")
    
    # Copy Mastercard traces
    mc_subdir = training_input / "Mastercard"
    mc_subdir.mkdir(exist_ok=True)
    for trace_file in traces['mastercard']:
        link_path = mc_subdir / trace_file.name
        if link_path.exists():
            link_path.unlink()
        try:
            # Try symlink first (faster)
            os.symlink(trace_file, link_path)
            print(f"  [OK] Linked: {trace_file.name}")
        except (OSError, NotImplementedError):
            # Fallback to copy if symlink not supported
            shutil.copy2(trace_file, link_path)
            print(f"  [OK] Copied: {trace_file.name}")
    
    # Copy Greenvisa traces (these don't have labels, so they'll use external labels)
    gv_subdir = training_input / "Greenvisa"
    gv_subdir.mkdir(exist_ok=True)
    for trace_file in traces['greenvisa']:
        link_path = gv_subdir / trace_file.name
        if link_path.exists():
            link_path.unlink()
        try:
            os.symlink(trace_file, link_path)
            print(f"  [OK] Linked: {trace_file.name}")
        except (OSError, NotImplementedError):
            shutil.copy2(trace_file, link_path)
            print(f"  [OK] Copied: {trace_file.name}")
    
    print(f"\n[OK] Training input ready at: {training_input}")
    return training_input

def main():
    print_header("3DES TRAINING DATA PREPARATION")
    
    print("""
This script prepares training data for 3DES models:
  [OK] Collects Mastercard 3DES traces (with labels)
  [OK] Collects Greenvisa 3DES traces (without native labels)
  [OK] Creates symlinks to avoid copying large files
  [OK] Prepares directory structure for retraining
""")
    
    # Step 1: Collect traces
    traces = collect_3des_traces()
    
    if not traces['mastercard']:
        print("\n✗ ERROR: No Mastercard 3DES traces found!")
        print("   Check: I:\\freelance\\SCA-Smartcard-Pipeline-3\\Input1\\Mastercard")
        return 1
    
    # Step 2: Prepare training input directory
    training_input = prepare_training_input_dir(traces)
    
    # Step 3: Summary
    print_header("NEXT STEPS")
    
    print(f"""
✓ Data preparation complete!

Training will use:
  ├─ Mastercard 3DES: {len(traces['mastercard'])} traces (with labels from KALKI file)
  └─ Greenvisa 3DES: {len(traces['greenvisa'])} traces (with labels from external map lookup)

Run retraining with:
  python retrain_3des_with_filtered_data.py \\
    --input_dir "{training_input}" \\
    --backup \\
    --epochs 200

Location of prepared data:
  {training_input}
  ├─ Mastercard/
  │  └─ *.csv (Mastercard 3DES traces)
  └─ Greenvisa/
     └─ *.csv (Greenvisa 3DES traces)
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
