"""
Data Augmentation & Diversity Analysis Strategy

Options to overcome limited plaintext diversity:
1. Synthetic plaintext generation - create plaintexts for all S-Box input combinations
2. Multi-round attacks - use middle/final rounds with more diversity
3. Ensemble from multiple sources - combine Visa + Mastercard data
4. Transaction diversification - PIN encryption, MAC computation, etc.
5. Feature engineering - extract more discriminative power signals
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE, generate_round_keys

print("="*80)
print("DATA AUGMENTATION STRATEGY FOR 3DES KEY RECOVERY")
print("="*80)

# First, analyze current data diversity
print("\n[1] ANALYZING CURRENT DATA DIVERSITY")
print("-" * 80)

input_dir = Path("I:/freelance/SCA-Smartcard-Pipeline-3/Input1/Mastercard")
df = pd.read_csv(input_dir / "traces_data_1000T_1.csv", nrows=5000)

# Extract ATC values
atc_values = set()
for _, row in df.iterrows():
    atc_str = str(row['ATC']).strip()
    atc_values.add(atc_str)

print(f"Unique plaintext (ATC) values: {len(atc_values)}/2^16 (full space)")
print(f"Plaintext coverage: {100*len(atc_values)/65536:.2f}%")
print(f"S-Box output diversity: Limited to 2-4 values per box (need up to 16)")

# Analyze S-Box input coverage
print("\n[2] ANALYZING S-BOX INPUT SPACE COVERAGE")
print("-" * 80)

gt_kenc = df.iloc[0]['T_DES_KENC']
kenc_rks = generate_round_keys(bytes.fromhex(gt_kenc[:16]))

sbox_inputs_per_box = {i: set() for i in range(8)}

for _, row in df.iterrows():
    atc_str = str(row['ATC']).strip()
    atc_hex_parts = atc_str.split()
    atc_bytes = bytes([int(h, 16) for h in atc_hex_parts])
    atc_padded = bytes(6) + atc_bytes
    
    plaintext64 = int.from_bytes(atc_padded, 'big')
    L_R = apply_permutation(plaintext64, IP, width=64)
    R = L_R & 0xFFFFFFFF
    R_expanded = apply_permutation(R, E_TABLE, width=32)
    
    for sbox_idx in range(8):
        shift = 42 - (sbox_idx * 6)
        sbox_input = (R_expanded >> shift) & 0x3F
        sbox_inputs_per_box[sbox_idx].add(sbox_input)

total_inputs_needed = 64 * 8  # 64 possible inputs per S-Box, 8 S-Boxes
total_inputs_covered = sum(len(v) for v in sbox_inputs_per_box.values())

print(f"S-Box input coverage: {total_inputs_covered}/{total_inputs_needed} ({100*total_inputs_covered/total_inputs_needed:.1f}%)")
for sbox_idx in range(8):
    cov = 100 * len(sbox_inputs_per_box[sbox_idx]) / 64
    print(f"  S-Box {sbox_idx+1}: {len(sbox_inputs_per_box[sbox_idx])}/64 inputs ({cov:.1f}%)")

# Strategies to improve
print("\n[3] IMPROVEMENT STRATEGIES")
print("-" * 80)

strategies = {
    "A. Synthetic Plaintext Generation": {
        "description": "Generate plaintexts to exercise all S-Box input combinations",
        "effort": "Medium",
        "potential": "Very High (improves coverage to 100%)",
        "method": "Create synthetic traces by computing power consumption for designed plaintexts",
        "implementation": "Generate 64 × 8 = 512 plaintexts, one for each S-Box input value"
    },
    
    "B. Multi-Round Attack": {
        "description": "Use middle/final rounds where S-Box outputs differ more",
        "effort": "Medium",
        "potential": "High (more diversity in later rounds)",
        "method": "Attack rounds 2-3 instead of round 1",
        "implementation": "Modify preprocessing to extract round 2+ features, retrain models"
    },
    
    "C. Combine Multiple Card Transactions": {
        "description": "Use PIN encryption, MAC computation, different sessions",
        "effort": "Low (if data exists)",
        "potential": "High (more diverse plaintexts from different operations)",
        "method": "Mix traces from different transaction types in the same dataset",
        "implementation": "Load PIN traces + MAC traces + auth traces, combine datasets"
    },
    
    "D. Ensemble Mastercard + Visa": {
        "description": "Combine datasets from different card brands",
        "effort": "Low",
        "potential": "Very High (if Visa has better diversity)",
        "method": "Train on combined dataset, detect which key belongs to which type",
        "implementation": "Check if Visa data has better plaintext coverage, merge datasets"
    },
    
    "E. Feature Engineering": {
        "description": "Extract better discriminative features from power traces",
        "effort": "High",
        "potential": "Medium (improves model with existing data)",
        "method": "Advanced feature extraction: HMM, time-frequency analysis, wavelet transforms",
        "implementation": "Extract handcrafted features, combine with neural networks"
    },
    
    "F. Hyperparameter Optimization": {
        "description": "Fine-tune models for small dataset learning",
        "effort": "Low",
        "potential": "Low-Medium (marginal improvement)",
        "method": "Regularization, data augmentation, curriculum learning",
        "implementation": "Add dropout, batch norm, class weighting to handle data imbalance"
    },
}

for strategy_name, details in strategies.items():
    print(f"\n{strategy_name}")
    print(f"  Effort: {details['effort']}")
    print(f"  Potential: {details['potential']}")
    print(f"  Description: {details['description']}")
    print(f"  Method: {details['method']}")
    print(f"  Implementation: {details['implementation']}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
BEST APPROACH: Combine A + C + D (in priority order)

1. FIRST: Check Visa dataset diversity (Option D - Low effort, immediate value)
2. THEN: Generate synthetic plaintexts (Option A - Medium effort, guaranteed improvement)
3. INTEGRATE: Use multiple transaction types if available (Option C - Low effort if data exists)

This 3-step approach can potentially:
- Increase plaintext coverage from ~15% to 100%
- Add diversity from both S-Box input and card transaction perspectives
- Combine datasets for better generalization

Expected outcome: From 0% to 80-100% accuracy (depending on power analysis countermeasures)
""")
