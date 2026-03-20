#!/usr/bin/env python3
"""
Analyze predictions to understand what went wrong.
Check consistency and test key slot permutations.
"""

import sys
sys.path.insert(0, 'pipeline-code')

import pandas as pd
import numpy as np
from itertools import permutations

# Load the metadata with ground truth
print("Loading metadata...")
meta = pd.read_csv('Output/mastercard_processed/3des/Y_meta.csv')

# Ground truth keys (from CSV)
ground_truth = {
    'KENC': 'T_DES_KENC',
    'KMAC': 'T_DES_KMAC', 
    'KDEK': 'T_DES_KDEK'
}

# Extract unique ground truth
gt_kenc = meta.iloc[0]['T_DES_KENC']
gt_kmac = meta.iloc[0]['T_DES_KMAC']
gt_kdek = meta.iloc[0]['T_DES_KDEK']

print(f"\n{'='*80}")
print("GROUND TRUTH KEYS (from Mastercard CSV)")
print(f"{'='*80}")
print(f"KENC: {gt_kenc}")
print(f"KMAC: {gt_kmac}")
print(f"KDEK: {gt_kdek}")

# Load predictions from inference output
print(f"\n{'='*80}")
print("CHECKING FOR PREDICTIONS IN OUTPUT FILES")
print(f"{'='*80}")

output_files = [
    'Output/mastercard_test/accuracy_summary.csv',
    'Output/mastercard_processed/3des/Y_meta.csv'
]

for f in output_files:
    try:
        df = pd.read_csv(f)
        cols = list(df.columns)
        print(f"\n{f}:")
        print(f"  Columns: {cols}")
        print(f"  Rows: {len(df)}")
        
        # Check for prediction columns
        pred_cols = [c for c in cols if 'pred' in c.lower() or '3des' in c.lower()]
        if pred_cols:
            print(f"  Prediction columns found: {pred_cols}")
    except Exception as e:
        print(f"\n{f}: {e}")

# Try to run inference again and capture predictions
print(f"\n{'='*80}")
print("RUNNING INFERENCE TO CAPTURE PREDICTIONS")
print(f"{'='*80}")

from src.inference_3des import recover_3des_keys

try:
    predicted = recover_3des_keys(
        processed_dir='Output/mastercard_processed',
        model_dir='pipeline-code/models',
        card_type='mastercard',
        n_attack=10000
    )
    
    print("\nPredictions from inference:")
    for key, value in predicted.items():
        print(f"  {key}: {value}")
    
    # Test key permutations
    print(f"\n{'='*80}")
    print("KEY PERMUTATION TESTING")
    print(f"{'='*80}")
    
    pred_kenc = predicted.get('3DES_KENC', '')
    pred_kmac = predicted.get('3DES_KMAC', '')
    pred_kdek = predicted.get('3DES_KDEK', '')
    
    print(f"\nPredicted keys:")
    print(f"  KENC: {pred_kenc}")
    print(f"  KMAC: {pred_kmac}")
    print(f"  KDEK: {pred_kdek}")
    
    # Check all 6 permutations of key assignments
    key_slots = ['KENC', 'KMAC', 'KDEK']
    pred_vals = [pred_kenc, pred_kmac, pred_kdek]
    gt_vals = [gt_kenc, gt_kmac, gt_kdek]
    
    print(f"\nTesting all permutations of key slot assignments:")
    matches = []
    for perm in permutations(range(3)):
        # perm[0] = which predicted key goes to KENC slot
        # perm[1] = which predicted key goes to KMAC slot
        # perm[2] = which predicted key goes to KDEK slot
        test_kenc = pred_vals[perm[0]]
        test_kmac = pred_vals[perm[1]]
        test_kdek = pred_vals[perm[2]]
        
        match_count = 0
        if test_kenc and test_kenc.upper() == gt_kenc.upper():
            match_count += 1
        if test_kmac and test_kmac.upper() == gt_kmac.upper():
            match_count += 1
        if test_kdek and test_kdek.upper() == gt_kdek.upper():
            match_count += 1
        
        perm_str = f"{key_slots[perm[0]]}->{key_slots[0]}, {key_slots[perm[1]]}->{key_slots[1]}, {key_slots[perm[2]]}->{key_slots[2]}"
        print(f"  {perm_str}: {match_count}/3 matches")
        
        if match_count > 0:
            matches.append((perm, match_count, perm_str))
    
    if matches:
        print(f"\n>>> Best permutation: {matches[0][2]} ({matches[0][1]}/3 matches)")
    else:
        print("\n>>> No key slot permutation matches")
    
    # Check consistency: Are all 10,000 traces predicting the same key?
    print(f"\n{'='*80}")
    print("CHECKING PREDICTION CONSISTENCY")
    print(f"{'='*80}")
    print("\nFor consistency check, need individual trace predictions.")
    print("Look for per-trace prediction files in Output/mastercard_processed/3des/")
    
    import os
    pred_files = [f for f in os.listdir('Output/mastercard_processed/3des/') if 'pred' in f.lower()]
    if pred_files:
        print(f"\nFound prediction files:")
        for f in pred_files:
            print(f"  - {f}")
    else:
        print("\nNo per-trace prediction files found.")
        print("Models may only produce aggregate predictions, not per-trace.")
    
except Exception as e:
    print(f"\nError during inference: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
