"""
Root cause analysis: check why only 4 S-Box classes exist
"""
import pandas as pd
import sys
sys.path.insert(0, "I:\\freelance\\SCA Smartcard ML Pipeline-3des\\pipeline-code")

from src.crypto import generate_round_keys, apply_permutation, IP, E_TABLE, des_sbox_output

# Load metadata
meta = pd.read_csv('I:\\freelance\\SCA Smartcard ML Pipeline-3des\\3des-pipeline\\Processed\\3des\\Y_meta.csv')

print("=== ATC ANALYSIS ===")
for i in range(8):
    col = f'ATC_{i}'
    if col in meta.columns:
        unique_atc = meta[col].nunique()
        print(f'{col}: {unique_atc} unique values')

# Get ground truth key
k_hex = meta.iloc[0]['T_DES_KENC']
print(f'\nGround truth KENC K1: {k_hex[:16]}')
key_bytes = bytes.fromhex(k_hex[:16])
round_keys = generate_round_keys(key_bytes)
rk1 = round_keys[0]

# Compute expected S-Box outputs for S-Box 1
print('\n=== EXPECTED SBOX1 OUTPUTS ===')
sbox_idx = 0  # S-Box 1 (0-indexed)

# For each unique ATC row, compute what S-Box output should be
atc_values = meta[['ATC_0', 'ATC_1', 'ATC_2', 'ATC_3', 'ATC_4', 'ATC_5', 'ATC_6', 'ATC_7']].drop_duplicates()
print(f'Unique ATC rows: {len(atc_values)}')

outputs = []
for idx, row in atc_values.iterrows():
    try:
        atc_bytes = bytes([int(row[f'ATC_{i}']) for i in range(8)])
        block_int = int.from_bytes(atc_bytes, 'big')
        plaintext_permuted = apply_permutation(block_int, IP, width=64)
        r0 = plaintext_permuted & 0xFFFFFFFF
        r0_expanded = apply_permutation(r0, E_TABLE, width=32)
        xor_result = r0_expanded ^ rk1
        shift = 42 - (sbox_idx * 6)
        six_bit_input = (xor_result >> shift) & 0x3F
        sbox_output = des_sbox_output(sbox_idx, six_bit_input)
        outputs.append(sbox_output)
        if idx < 10:
            print(f'ATC={atc_bytes.hex()}: SBOX1 output = {sbox_output}')
    except Exception as e:
        print(f'Error for ATC row {idx}: {e}')

print(f'\nAll unique S-Box outputs: {sorted(set(outputs))}')
print(f'Number of unique outputs: {len(set(outputs))}')
print(f'Output distribution: {[(o, outputs.count(o)) for o in sorted(set(outputs))]}')
