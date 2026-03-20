"""
Synthetic Plaintext Generation for 100% S-Box Input Coverage

Strategy: Generate plaintexts that force each S-Box to take all 64 possible input values,
effectively creating 512+ synthetic training samples from first principles.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import (
    des_sbox_output, apply_permutation, IP, E_TABLE,
    generate_round_keys
)

def generate_synthetic_plaintexts_for_sbox(sbox_idx, round_key_int):
    """
    Generate plaintexts that force a specific S-Box to take all 64 possible input values
    
    For each desired S-Box output:
    1. Work backwards: choose a target S-Box input (0-63)
    2. Design plaintext bits to achieve that input after XOR with key
    3. Keep other S-Boxes neutral (0 input)
    
    Args:
        sbox_idx: Which S-Box (0-7)
        round_key_int: 48-bit round key
    
    Returns:
        plaintexts: List of 8-byte plaintexts that exercise all S-Box inputs for this box
    """
    
    plaintexts = []
    
    # For each possible 6-bit input (0-63) to this S-Box
    for target_sbox_input in range(64):
        # We want: (R_expanded >> shift) & 0x3F = target_sbox_input
        # Working backwards from DES structure...
        
        # Create a simple test plaintext
        # In a full implementation, we'd solve for exact plaintext bits
        # For now, use a practical approach: vary specific plaintext bits
        
        # The plaintext after IP becomes L0||R0
        # We control R0 (right half)
        # After expansion and XOR with key in first S-Box:
        # We need bits to align with target_sbox_input
        
        # Simple approach: use bits directly
        for test_val in range(256):
            pt_bytes = bytearray(8)
            # Use different bytes to test different plaintext variations
            pt_bytes[0] = test_val
            pt_bytes[1] = target_sbox_input
            pt_bytes[2] = (target_sbox_input >> 2) & 0xFF
            
            plaintext64 = int.from_bytes(bytes(pt_bytes), 'big')
            L_R = apply_permutation(plaintext64, IP, width=64)
            R = L_R & 0xFFFFFFFF
            R_expanded = apply_permutation(R, E_TABLE, width=32)
            shift = 42 - (sbox_idx * 6)
            actual_input = ((R_expanded >> shift) ^ (round_key_int >> shift)) & 0x3F
            
            if actual_input == target_sbox_input:
                plaintexts.append(bytes(pt_bytes))
                break
    
    return plaintexts

def generate_synthetic_dataset(ground_truth_kenc, num_repeats=5):
    """
    Generate a complete synthetic dataset with 100% S-Box input coverage
    
    Strategy:
    1. Generate plaintexts for all 64 inputs per S-Box
    2. Repeat with different trace "noise" patterns (num_repeats times)
    3. Create synthetic power traces using power model
    
    Args:
        ground_truth_kenc: KENC key in hex
        num_repeats: How many times to repeat each plaintext with variations
    
    Returns:
        synthetic_traces: Synthetic power measurements
        synthetic_atc: Corresponding plaintext bytes
        synthetic_labels: S-Box outputs for each trace
    """
    
    print("Generating synthetic plaintexts...")
    
    kenc_rks = generate_round_keys(bytes.fromhex(ground_truth_kenc[:16]))
    rk1 = kenc_rks[0]
    
    # Generate plaintexts that exercise all S-Box inputs
    all_plaintexts = []
    all_labels = []
    
    for sbox_idx in range(8):
        print(f"  S-Box {sbox_idx+1}: Generating plaintexts for all 64 inputs...")
        
        for target_input in range(64):
            # Find a plaintext that produces this S-Box input
            found = False
            for test_byte in range(256):
                pt_bytes = bytearray(8)
                pt_bytes[sbox_idx] = test_byte
                pt_bytes[7 - sbox_idx % 4] = target_input
                
                plaintext64 = int.from_bytes(bytes(pt_bytes), 'big')
                L_R = apply_permutation(plaintext64, IP, width=64)
                R = L_R & 0xFFFFFFFF
                R_expanded = apply_permutation(R, E_TABLE, width=32)
                shift = 42 - (sbox_idx * 6)
                actual_input = ((R_expanded >> shift) ^ (rk1 >> shift)) & 0x3F
                sbox_output = des_sbox_output(sbox_idx, actual_input)
                
                if actual_input == target_input:
                    # Store this plaintext
                    all_plaintexts.append(bytes(pt_bytes))
                    all_labels.append({
                        'sbox_idx': sbox_idx,
                        'sbox_input': target_input,
                        'sbox_output': sbox_output
                    })
                    found = True
                    break
            
            if not found and target_input % 16 == 0:
                print(f"    Warning: Could not find plaintext for S-Box {sbox_idx} input {target_input}")
    
    print(f"\nGenerated {len(all_plaintexts)} unique plaintexts")
    
    # Create synthetic traces with power model
    print("\nGenerating synthetic power traces...")
    
    synthetic_traces = []
    synthetic_labels_per_sbox = {i: [] for i in range(8)}
    
    for plaintext_bytes in all_plaintexts:
        # Compute ALL S-Box outputs for this plaintext
        labels = {}
        
        plaintext64 = int.from_bytes(plaintext_bytes, 'big')
        L_R = apply_permutation(plaintext64, IP, width=64)
        R = L_R & 0xFFFFFFFF
        R_expanded = apply_permutation(R, E_TABLE, width=32)
        
        for sbox_idx in range(8):
            shift = 42 - (sbox_idx * 6)
            sbox_input = ((R_expanded >> shift) ^ (rk1 >> shift)) & 0x3F
            sbox_output = des_sbox_output(sbox_idx, sbox_input)
            labels[sbox_idx] = sbox_output
            synthetic_labels_per_sbox[sbox_idx].append(sbox_output)
        
        # Generate synthetic trace using Hamming weight power model + noise
        num_samples = 2200
        trace = np.zeros(num_samples, dtype=np.float32)
        
        # Power model: Hamming weight of all intermediate values
        for sbox_idx in range(8):
            hw = bin(labels[sbox_idx]).count('1')
            # Distribute power across trace samples
            peak_location = 200 + sbox_idx * 250  # Spread peaks across trace
            width = 50
            for t in range(max(0, peak_location - width), min(num_samples, peak_location + width)):
                gauss = np.exp(-((t - peak_location)**2) / (2 * width**2))
                trace[t] += 0.1 * hw * gauss
        
        # Add noise
        noise = np.random.normal(0, 0.02, num_samples)
        trace += noise
        
        synthetic_traces.append(trace)
    
    # Repeat with variations
    print(f"Repeating traces {num_repeats} times with noise variations...")
    
    final_traces = []
    final_labels = {}
    
    for repeat_idx in range(num_repeats):
        for pt_idx, plaintext_bytes in enumerate(all_plaintexts):
            # Add repeated trace with noise variation
            plaintext64 = int.from_bytes(plaintext_bytes, 'big')
            L_R = apply_permutation(plaintext64, IP, width=64)
            R = L_R & 0xFFFFFFFF
            R_expanded = apply_permutation(R, E_TABLE, width=32)
            
            trace = np.zeros(2200, dtype=np.float32)
            for sbox_idx in range(8):
                shift = 42 - (sbox_idx * 6)
                sbox_input = ((R_expanded >> shift) ^ (rk1 >> shift)) & 0x3F
                sbox_output = des_sbox_output(sbox_idx, sbox_input)
                hw = bin(sbox_output).count('1')
                peak_location = 200 + sbox_idx * 250
                width = 50
                for t in range(max(0, peak_location - width), min(2200, peak_location + width)):
                    gauss = np.exp(-((t - peak_location)**2) / (2 * width**2))
                    trace[t] += 0.1 * hw * gauss
            
            noise = np.random.normal(0, 0.03 * (1 + repeat_idx / num_repeats), 2200)
            trace += noise
            
            final_traces.append(trace)
            
            # Store labels
            if pt_idx not in final_labels:
                final_labels[pt_idx] = {}
            final_labels[pt_idx][repeat_idx] = {
                'plaintext': plaintext_bytes.hex(),
                'sbox_outputs': {}
            }
            
            for sbox_idx in range(8):
                shift = 42 - (sbox_idx * 6)
                sbox_input = ((R_expanded >> shift) ^ (rk1 >> shift)) & 0x3F
                sbox_output = des_sbox_output(sbox_idx, sbox_input)
                final_labels[pt_idx][repeat_idx]['sbox_outputs'][sbox_idx] = int(sbox_output)
    
    final_traces = np.array(final_traces, dtype=np.float32)
    
    print(f"\nSynthetic dataset generated:")
    print(f"  Total synthetic traces: {len(final_traces)}")
    print(f"  Trace shape: {final_traces.shape}")
    print(f"  Coverage: 64 inputs × 8 S-Boxes × {num_repeats} repeats = ~{64 * 8 * num_repeats} samples")
    
    return final_traces, final_labels

def main():
    print("="*80)
    print("SYNTHETIC PLAINTEXT GENERATION FOR DATA AUGMENTATION")
    print("="*80)
    
    # Load ground truth key
    input_dir = Path("I:/freelance/SCA-Smartcard-Pipeline-3/Input1/Mastercard")
    df = pd.read_csv(input_dir / "traces_data_1000T_1.csv", nrows=1)
    
    gt_kenc = df.iloc[0]['T_DES_KENC']
    print(f"\nGround Truth KENC: {gt_kenc}")
    
    # Generate synthetic data
    synthetic_traces, synthetic_labels = generate_synthetic_dataset(gt_kenc, num_repeats=3)
    
    # Save synthetic data
    print("\n" + "="*80)
    print("SAVING SYNTHETIC DATA")
    print("="*80)
    
    output_dir = Path("Output/synthetic_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "synthetic_traces.npy", synthetic_traces)
    
    # Save labels as CSV for inspection
    labels_list = []
    for pt_idx, variants in synthetic_labels.items():
        for repeat_idx, data in variants.items():
            row = {
                'plaintext': data['plaintext'],
            }
            for sbox_idx in range(8):
                row[f'sbox_{sbox_idx}_output'] = data['sbox_outputs'][sbox_idx]
            labels_list.append(row)
    
    labels_df = pd.DataFrame(labels_list)
    labels_df.to_csv(output_dir / "synthetic_labels.csv", index=False)
    
    print(f"Saved synthetic_traces.npy: {synthetic_traces.shape}")
    print(f"Saved synthetic_labels.csv: {labels_df.shape}")
    
    # Analyze coverage
    print("\n" + "="*80)
    print("COVERAGE ANALYSIS")
    print("="*80)
    
    for sbox_idx in range(8):
        outputs = labels_df[f'sbox_{sbox_idx}_output'].unique()
        print(f"S-Box {sbox_idx+1}: {len(outputs)}/16 possible outputs covered = {100*len(outputs)/16:.1f}%")

if __name__ == '__main__':
    main()
