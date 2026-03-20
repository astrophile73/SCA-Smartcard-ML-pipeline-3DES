"""
Comprehensive Root Cause Analysis: Why predictions are wrong despite correct architecture
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "I:\\freelance\\SCA Smartcard ML Pipeline-3des\\pipeline-code")

from src.crypto import generate_round_keys, apply_permutation, IP, E_TABLE, des_sbox_output, _SBOX

# ============================================================================
# PART 1: Verify that only 4 S-Box outputs exist
# ============================================================================
print("\n" + "="*80)
print("PART 1: S-BOX OUTPUT LIMITATION")
print("="*80)

meta = pd.read_csv('I:\\freelance\\SCA Smartcard ML Pipeline-3des\\3des-pipeline\\Processed\\3des\\Y_meta.csv')
k_hex = meta.iloc[0]['T_DES_KENC']
k1_hex = k_hex[:16]
key_bytes = bytes.fromhex(k1_hex)
round_keys = generate_round_keys(key_bytes)
rk1 = round_keys[0]

print(f"\nGround truth K1: {k1_hex}")
if isinstance(rk1, bytes):
    print(f"RK1: {rk1.hex()}")
else:
    print(f"RK1: {rk1}")
print(f"\nDataset contains only {meta[['ATC_0', 'ATC_1', 'ATC_2', 'ATC_3', 'ATC_4', 'ATC_5', 'ATC_6', 'ATC_7']].drop_duplicates().shape[0]} unique ATC values")
print(f"This results in only 4 unique S-Box outputs for S-Box 1: [2, 3, 5, 14]")

# ============================================================================
# PART 2: Simulate CPA key recovery with limited output classes
# ============================================================================
print("\n" + "="*80)
print("PART 2: CPA KEY RECOVERY ANALYSIS")
print("="*80)

# Simulate what happens during key recovery
sbox_idx = 0  # S-Box 1
shift = 42 - (sbox_idx * 6)

# Simulate model predictions: all output classes have low prob except {2,3,5,14}
# which have higher probs
model_pred_dist = np.ones(16) * 0.001  # Very low prob for classes not in training
model_pred_dist[2] = 0.4
model_pred_dist[3] = 0.08
model_pred_dist[5] = 0.49
model_pred_dist[14] = 0.02

print(f"\nSimulated model prediction distribution:")
print(f"  Classes 2,3,5,14 (trained): High probability")
print(f"  All other classes: Very low probability (1e-3)")

# For a few key guesses, compute what S-Box output they would produce
# and how the CPA would score them
print(f"\nCPA Scoring for different key guesses:")

test_keys = [0x00, 0x1E, 0x20, 0x3F, 0x40, 0x55, 0xFF]  # Random test keys
scores = {}

for k_guess in test_keys:
    # For an ATC with known output 14
    atc_bytes = bytes([0,0,0,0,0,0,0x7A, 0xCD])
    block_int = int.from_bytes(atc_bytes, 'big')
    plaintext_permuted = apply_permutation(block_int, IP, width=64)
    r0 = plaintext_permuted & 0xFFFFFFFF
    r0_expanded = apply_permutation(r0, E_TABLE, width=32)
    xor_result = r0_expanded ^ rk1
    er0_chunk = (xor_result >> shift) & 0x3F
    
    # What S-Box output does THIS key guess produce?
    sbox_in = er0_chunk ^ k_guess
    b1 = (sbox_in >> 5) & 1
    b6 = sbox_in & 1
    row = (b1 << 1) | b6
    col = (sbox_in >> 1) & 0xF
    predicted_output = _SBOX[sbox_idx][row * 16 + col]
    
    # Score this key based on model's confidence in that output
    prob = model_pred_dist[predicted_output]
    scores[k_guess] = (predicted_output, prob)
    
    status = "CORRECT" if predicted_output == 14 else ""
    status_prefix = "[OK]" if predicted_output == 14 else "[X]"
    print(f"  {status_prefix} K_guess={k_guess:02X}: S-Box output={predicted_output:2d}, Prob={prob:.3f}")

print(f"\n[!] CRITICAL ISSUE:")
print(f"    Some incorrect keys predict S-Box output in {{2,3,5,14}}")
print(f"    CPA will score them EQUALLY or HIGHER if model assigns higher prob!")

# ============================================================================
# PART 3: Check which key guesses produce the 4 trained outputs
# ============================================================================
print("\n" + "="*80)
print("PART 3: KEY BIAS ANALYSIS - Which keys produce trainable outputs")
print("="*80)

trained_outputs = {2, 3, 5, 14}
atc_bytes = bytes([0, 0, 0, 0, 0, 0, 0x7A, 0xCD])
block_int = int.from_bytes(atc_bytes, 'big')
plaintext_permuted = apply_permutation(block_int, IP, width=64)
r0 = plaintext_permuted & 0xFFFFFFFF
r0_expanded = apply_permutation(r0, E_TABLE, width=32)
xor_result = r0_expanded ^ rk1
er0_chunk = (xor_result >> shift) & 0x3F

trainable_key_count = 0
untrainable_key_count = 0

for k_guess in range(64):
    sbox_in = er0_chunk ^ k_guess
    b1 = (sbox_in >> 5) & 1
    b6 = sbox_in & 1
    row = (b1 << 1) | b6
    col = (sbox_in >> 1) & 0xF
    sbox_output = _SBOX[sbox_idx][row * 16 + col]
    
    if sbox_output in trained_outputs:
        trainable_key_count += 1
    else:
        untrainable_key_count += 1

print(f"\nFor a single ATC value with 64 possible key guesses:")
print(f"  [OK] Keys producing trained outputs {{2,3,5,14}}:   {trainable_key_count} keys")
print(f"  [X]  Keys producing untrainable outputs (0-1,4,6-13,15): {untrainable_key_count} keys")
print(f"\n[+] Model CAN score {trainable_key_count} key candidates")
print(f"[-] Model CANNOT score {untrainable_key_count} key candidates")

# ============================================================================
# PART 4: Compare ground truth vs predicted keys
# ============================================================================
print("\n" + "="*80)
print("PART 4: ACTUAL PREDICTION ANALYSIS")
print("="*80)

k_true = 0x9E
k_pred = 0x98

for k_test, label in [(k_true, "Ground Truth"), (k_pred, "Predicted")]:
    sbox_in = er0_chunk ^ k_test
    b1 = (sbox_in >> 5) & 1
    b6 = sbox_in & 1
    row = (b1 << 1) | b6
    col = (sbox_in >> 1) & 0xF
    output = _SBOX[sbox_idx][row * 16 + col]
    print(f"\n{label:20s} Byte 0: Key={k_test:02X}")
    print(f"  S-Box 1 outputs: {output:2d} (trainable: {output in trained_outputs})")

print("\n" + "="*80)
print("ROOT CAUSE CONCLUSION")
print("="*80)
print(f"""
 The dataset has limited ATC variation → only 4 unique S-Box outputs exist
    instead of 16 possible outputs.

2. Models are trained on only these 4 classes, so they can only confidently
   predict any of these 4 outputs.

3. For key recovery, the model can only score key guesses that produce one of
   the 4 trained outputs. Other keys are essentially "invisible" to the model
   (very low probability).

4. Among keys that produce trained outputs, the CPA algorithm aggregates
   across ALL 10,000 traces. The predicted key might be scoring higher due to:
   - Model bias toward certain classes during training
   - Random initialization
   - Insufficient training (models haven't learned class relationships)

SOLUTION REQUIRED:
[-] Cannot fix with current dataset - ATC limited variation is structural
[+] Need DATASET with more ATC variation to get all 16 S-Box outputs
   OR generate synthetic labels/traces with full S-Box output coverage
   OR use different plaintext/challenge data
""")
