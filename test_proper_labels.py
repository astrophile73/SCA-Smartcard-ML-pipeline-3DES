#!/usr/bin/env python3
"""
Proper way to load external labels and fix key extraction.
"""

import sys
import os

sys.path.insert(0, "pipeline-code")

from src.ingest import TraceDataset
from src.external_label_map import load_external_3des_label_map
from src.gen_labels import compute_labels

KALKI_FILE = "KALKi TEST CARD.xlsx"
INPUT_DIR = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1"

print("=" * 80)
print("LOADING EXTERNAL LABELS PROPERLY")
print("=" * 80)

# Load using the proper loader
print(f"\nLoading KALKI file: {KALKI_FILE}")
external_label_map = load_external_3des_label_map(KALKI_FILE)

print(f"Label map keys: {list(external_label_map.keys())}")
for key, data in external_label_map.items():
    print(f"  {key}:")
    for k, v in data.items():
        print(f"    {k}: {v}")

# Now use TraceDataset with the external map
print(f"\n\n" + "=" * 80)
print("TESTING WITH PROPER EXTERNAL LABEL MAP")
print("=" * 80)

try:
    ds = TraceDataset(
        INPUT_DIR,
        file_pattern="*trace*.*",
        card_type="universal",
        external_label_map=external_label_map,
        strict_label_mode=False
    )
    meta = ds._load_metadata()
    
    print(f"\n[OK] Loaded {len(meta)} traces")
    
    # Check key extraction
    print(f"\nKey extraction results:")
    n_with_kenc = (meta["T_DES_KENC"].astype(str).str.strip() != "").sum()
    n_with_kmac = (meta["T_DES_KMAC"].astype(str).str.strip() != "").sum()
    n_with_kdek = (meta["T_DES_KDEK"].astype(str).str.strip() != "").sum()
    
    print(f"  T_DES_KENC: {n_with_kenc}/{len(meta)} ({100*n_with_kenc/len(meta):.1f}%)")
    print(f"  T_DES_KMAC: {n_with_kmac}/{len(meta)} ({100*n_with_kmac/len(meta):.1f}%)")
    print(f"  T_DES_KDEK: {n_with_kdek}/{len(meta)} ({100*n_with_kdek/len(meta):.1f}%)")
    
    # Test label computation
    print(f"\n=== Testing label computation ===")
    
    sample_indices = [0, 10, 100, 1000, 10000]
    valid_count = 0
    total_tested = 0
    
    for idx in sample_indices:
        if idx >= len(meta):
            continue
        
        try:
            sample_meta = meta.iloc[idx:idx+1].copy()
            labels = compute_labels(sample_meta, sbox_idx=0, key_col="T_DES_KENC", stage=1)
            
            track2 = meta.iloc[idx]["Track2"]
            kenc = meta.iloc[idx]["T_DES_KENC"]
            
            is_valid = labels[0] >= 0
            if is_valid:
                valid_count += 1
            total_tested += 1
            
            status = "[OK]" if is_valid else "[FAIL]"
            print(f"  Row {idx:5d}: {status} Track2={str(track2)[:20]:20} Key={str(kenc)[:20]:20} Label={labels[0]}")
            
        except Exception as e:
            print(f"  Row {idx:5d}: [ERROR] {str(e)[:50]}")
    
    print(f"\nLabel computation success: {valid_count}/{total_tested}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
