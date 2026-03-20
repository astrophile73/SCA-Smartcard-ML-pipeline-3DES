"""
Corrected Diagnostic: Compare Training Data S-Box Outputs vs Ground Truth

Loads labels from .npy files and compares with expected outputs from ground truth keys.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE, generate_round_keys

def compute_expected_sbox_outputs(atc_bytes, round_key_int, sbox_idx):
    """Compute expected S-Box output for given plaintext and key"""
    plaintext64 = int.from_bytes(atc_bytes, 'big')
    L_R = apply_permutation(plaintext64, IP, width=64)
    R = L_R & 0xFFFFFFFF
    R_expanded = apply_permutation(R, E_TABLE, width=32)
    shift = 42 - (sbox_idx * 6)
    sbox_input = ((R_expanded >> shift) ^ (round_key_int >> shift)) & 0x3F
    return des_sbox_output(sbox_idx, sbox_input)

print("="*80)
print("DIAGNOSTIC: TRAINING DATA S-BOX OUTPUT MISMATCH")
print("="*80)

# Load ground truth
print("\n[1] Loading ground truth keys from raw traces...")
input_dir = Path("I:/freelance/SCA-Smartcard-Pipeline-3/Input1/Mastercard")
df_raw = pd.read_csv(input_dir / "traces_data_1000T_1.csv", nrows=1000)

gt_kenc = df_raw.iloc[0]['T_DES_KENC']
print(f"Ground Truth KENC: {gt_kenc}")

# Generate round keys
kenc_bytes = bytes.fromhex(gt_kenc[:16])
kenc_rks = generate_round_keys(kenc_bytes)
print(f"KENC Round Key 1: {kenc_rks[0]:012x}")

# Compute expected S-Box outputs
print("\n[2] Computing expected S-Box outputs from ground truth...")
expected_outputs = {}

for sbox_idx in range(8):
    outputs = set()
    for trace_idx in range(len(df_raw)):
        atc_str = str(df_raw.iloc[trace_idx]['ATC']).strip()
        atc_hex_parts = atc_str.split()
        atc_bytes = bytes([int(h, 16) for h in atc_hex_parts])
        atc_padded = bytes(6) + atc_bytes
        
        output = compute_expected_sbox_outputs(atc_padded, kenc_rks[0], sbox_idx)
        outputs.add(output)
    
    expected_outputs[sbox_idx] = sorted(outputs)
    print(f"  S-Box {sbox_idx+1}: {sorted(outputs)}")

# Load training labels from .npy files
print("\n[3] Loading training labels from .npy files...")
processed_dir = Path("Output/mastercard_processed")

training_outputs = {}
for sbox_idx in range(8):
    label_file = processed_dir / f"Y_labels_kenc_s1_sbox{sbox_idx+1}.npy"
    if label_file.exists():
        labels = np.load(label_file)
        training_outputs[sbox_idx] = sorted(np.unique(labels))[:20]  # Show first 20 unique
        print(f"  S-Box {sbox_idx+1}: {sorted(np.unique(labels))[:20]}")
    else:
        print(f"  S-Box {sbox_idx+1}: NOT FOUND ({label_file})")

# Compare
print("\n[4] COMPARISON RESULTS...")
print("\nKey: Expected outputs from ground truth vs Actual training labels\n")

for sbox_idx in range(8):
    exp = set(expected_outputs[sbox_idx])
    train = set(training_outputs.get(sbox_idx, []))
    
    match = "OK" if exp == train else "MISMATCH"
    print(f"S-Box {sbox_idx+1}: {match}")
    print(f"  Expected: {sorted(exp)}")
    print(f"  Training: {sorted(train)}")
    print()

# Summary
print("="*80)
print("SUMMARY & SOLUTION")
print("="*80)
print("""
If EXPECTED != TRAINING, this means:
- The training labels were generated with WRONG keys
- The preprocessing pipeline is not using the ground truth keys

SOLUTION:
You need to regenerate the training labels using the CORRECT ground truth keys.
The preprocessing function likely has a bug where it's not loading the keys properly.

Check:
1. The gen_labels_3des_fixed.py script
2. Make sure it's reading the ground truth keys from raw traces correctly
3. Verify it's using the same key for ALL traces in the dataset
4. Re-run preprocessing with the fix
""")
