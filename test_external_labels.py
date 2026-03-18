#!/usr/bin/env python3
"""
Load external label map from KALKI file and use it during key recovery.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, "pipeline-code")
from src.ingest import TraceDataset
from src.gen_labels import compute_labels

# Find and load KALKI label file
KALKI_FILE = "KALKi TEST CARD.xlsx"
INPUT_DIR = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1"

print("=" * 80)
print("LOADING EXTERNAL LABEL MAP")
print("=" * 80)

# Load KALKI file
if not os.path.exists(KALKI_FILE):
    # Try other locations
    possible_paths = [
        os.path.expanduser("~/KALKi TEST CARD.xlsx"),
        os.path.join(os.path.dirname(__file__), KALKI_FILE),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            KALKI_FILE = path
            break

print(f"\nLoading: {KALKI_FILE}")
df_labels = pd.read_excel(KALKI_FILE)

print(f"Loaded {len(df_labels)} label records")
print(f"Columns: {list(df_labels.columns)}")

# Create external label map
# Format: {profile: {Track2: {key_type: key_value}}}
external_label_map = {}

for idx, row in df_labels.iterrows():
    profile = str(row.get("PROFILE", "")).lower().strip()
    track2 = str(row.get("TRACK2", "")).strip()
    
    if not profile or not track2:
        continue
    
    kenc = str(row.get("3DES_KENC", "")).strip()
    kmac = str(row.get("3DES_KMAC", "")).strip()
    kdek = str(row.get("3DES_KDEK", "")).strip()
    
    if profile not in external_label_map:
        external_label_map[profile] = {}
    
    external_label_map[profile][track2] = {
        "T_DES_KENC": kenc,
        "T_DES_KMAC": kmac,
        "T_DES_KDEK": kdek,
    }
    
    print(f"\nLoaded {profile.upper()} / {track2[:20]}...")
    print(f"  KENC: {kenc}")
    print(f"  KMAC: {kmac}")
    print(f"  KDEK: {kdek}")

print(f"\n\nExternal label map summary:")
for profile, data in external_label_map.items():
    print(f"  {profile.upper()}: {len(data)} entries")

# Now test with TraceDataset using the external map
print(f"\n\n" + "=" * 80)
print("TESTING WITH EXTERNAL LABEL MAP")
print("=" * 80)

try:
    ds = TraceDataset(
        INPUT_DIR, 
        file_pattern="*trace*.*",
        card_type="universal",
        external_label_map=external_label_map,
        strict_label_mode=False  # Allow fallback to external map
    )
    meta = ds._load_metadata()
    
    print(f"\nLoaded {len(meta)} traces")
    
    # Check key extraction with external map
    print(f"\nKey extraction results with external map:")
    n_with_kenc = (meta["T_DES_KENC"].astype(str).str.strip() != "").sum()
    n_with_kmac = (meta["T_DES_KMAC"].astype(str).str.strip() != "").sum()
    n_with_kdek = (meta["T_DES_KDEK"].astype(str).str.strip() != "").sum()
    
    print(f"  Traces with KENC: {n_with_kenc}")
    print(f"  Traces with KMAC: {n_with_kmac}")
    print(f"  Traces with KDEK: {n_with_kdek}")
    
    # Check if labels compute successfully
    print(f"\nTesting label computation with external map:")
    sample_meta = meta.iloc[:5].copy()
    
    valid_count = 0
    for idx in range(len(sample_meta)):
        try:
            labels = compute_labels(sample_meta.iloc[idx:idx+1], sbox_idx=0, key_col="T_DES_KENC", stage=1)
            if labels[0] >= 0:
                valid_count += 1
                print(f"  Row {idx}: [OK] Label={labels[0]}")
            else:
                print(f"  Row {idx}: [FAIL] Label=-1")
        except Exception as e:
            print(f"  Row {idx}: [ERROR] {e}")
    
    print(f"\nValid labels: {valid_count} / {len(sample_meta)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
