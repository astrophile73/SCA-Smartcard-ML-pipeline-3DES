"""
3DES S-Box Input Label Generation (FIXED VERSION)

This module generates labels by extracting S-Box INPUTS (6-bit values, 0-63)
instead of S-Box OUTPUTS (4-bit values, 0-15).

S-Box input = XOR of expanded plaintext and round key
This directly contains key material and is better for CPA attack.

Key differences from gen_labels.py:
- Extract 6-bit S-Box INPUT instead of 4-bit S-Box OUTPUT
- Output values: 0-63 (6-bit, not 4-bit)
- Better information retention for key recovery
- Should result in more diverse label classes
"""

import numpy as np
import pandas as pd
try:
    from src.crypto import generate_round_keys, apply_permutation, IP, E_TABLE
except ImportError:
    from crypto import generate_round_keys, apply_permutation, IP, E_TABLE


def _parse_input_block(row) -> bytes:
    """Parse the input block (ATC) from metadata row"""
    def parse_atc_byte(val):
        try:
            return int(val)
        except Exception:
            try:
                return int(str(val), 16)
            except Exception:
                return 0

    # Prefer ATC_0..ATC_7 if available
    try:
        b = bytes([parse_atc_byte(row.get(f"ATC_{i}", 0)) for i in range(8)])
        if any(x != 0 for x in b):
            return b
    except Exception:
        pass

    # Fallback to 'ATC' raw field
    atc_raw = str(row.get("ATC", "")).replace(" ", "").strip()
    if atc_raw and atc_raw.lower() != "nan":
        if len(atc_raw) < 16:
            atc_raw = atc_raw.zfill(16)
        if len(atc_raw) > 16:
            atc_raw = atc_raw[:16]
        try:
            return bytes.fromhex(atc_raw)
        except Exception:
            pass

    return b"\x00" * 8


def compute_sbox_input_labels(metadata, sbox_idx, key_col="T_DES_KENC", stage: int = 1):
    """
    Computes S-Box INPUT labels (6-bit values, 0-63) for CPA attack.
    
    The S-Box input is: the 6-bit value that goes INTO the S-Box
    This is computed as: (expanded_plaintext_bits XOR round_key_bits)
    
    This is better than S-Box outputs for CPA because:
    1. It directly contains key material (XOR with plaintext)
    2. No information loss (output has only 4 bits)
    3. Better leakage correlation (Hamming Distance of 6-bit value)
    
    Args:
        metadata: DataFrame with columns [key_col, 'ATC_0', ..., 'ATC_7']
        sbox_idx: S-Box index (0-7, for S1-S8)
        key_col: Column name containing the 16-byte (32 hex) 3DES key
        stage: 1 targets K1/RK1 on input block; 2 targets K2/RK16
    
    Returns:
        numpy array of labels (6-bit integers, 0-63)
    """
    labels = []

    for idx, row in metadata.iterrows():
        try:
            # 1. Get Key
            k_hex = str(row.get(key_col, "")).strip()
            if not k_hex or k_hex.lower() == "nan" or all(c == "0" for c in k_hex):
                labels.append(-1)
                continue

            k_hex = k_hex.replace(" ", "").upper()
            if len(k_hex) < 32:
                k_hex = k_hex.zfill(32)
            if len(k_hex) > 32:
                k_hex = k_hex[:32]

            k1_hex = k_hex[:16]
            k2_hex = k_hex[16:32]

            input_block = _parse_input_block(row)
            
            # 2. Get Round Key
            if stage == 1:
                key_bytes = bytes.fromhex(k1_hex)
                round_keys = generate_round_keys(key_bytes)
                rk = round_keys[0]  # RK1 (48 bits)
            elif stage == 2:
                k2_bytes = bytes.fromhex(k2_hex)
                round_keys = generate_round_keys(k2_bytes)
                rk = round_keys[15]  # RK16 (48 bits)
            else:
                labels.append(-1)
                continue

            # 3. Compute S-Box Input
            # Apply Initial Permutation (IP) to plaintext
            block_int = int.from_bytes(input_block, "big")
            plaintext_permuted = apply_permutation(block_int, IP, width=64)
            
            # Extract right half (R0) - 32 bits
            r0 = plaintext_permuted & 0xFFFFFFFF
            
            # Expand R0 using E-table (32 -> 48 bits)
            r0_expanded = apply_permutation(r0, E_TABLE, width=32)
            
            # XOR with Round Key (this is the S-Box input in the DES design)
            xor_result = r0_expanded ^ rk  # This is 48 bits
            
            # Extract the 6-bit input for this specific S-Box
            # DES S-Box bit ordering: S1=bits 47-42, S2=bits 41-36, ..., S8=bits 5-0
            # Standard DES byte order: S1 uses bits 42-47 (from right), S2 uses 36-41, etc.
            shift = 42 - (sbox_idx * 6)
            six_bit_input = (xor_result >> shift) & 0x3F
            
            labels.append(six_bit_input)
            
        except Exception as e:
            labels.append(-1)
    
    return np.array(labels, dtype=np.int64)


def generate_sbox_input_labels(meta_path, sbox_idx, output_dir="Processed", key_col="T_DES_KENC", stage: int = 1):
    """
    Generates and saves S-Box INPUT labels for training.
    
    Args:
        meta_path: Path to metadata CSV
        sbox_idx: S-Box index (0-7 for S1-S8)
        output_dir: Directory to save labels
        key_col: Column containing the 3DES key
        stage: DES round stage (1 or 2)
    
    Returns:
        Path to saved labels file
    """
    import os
    
    # Load metadata
    df = pd.read_csv(meta_path, dtype={key_col: str, "Track2": str})
    
    # Compute S-Box INPUT labels
    labels = compute_sbox_input_labels(df, sbox_idx, key_col=key_col, stage=stage)
    
    # Save with clear naming
    key_suffix = key_col.replace("T_DES_", "").lower()
    output_path = os.path.join(output_dir, f"Y_labels_sbox_input_{key_suffix}_s{stage}_sbox{sbox_idx+1}.npy")
    np.save(output_path, labels)
    
    print(f"[OK] Saved S-Box input labels to {output_path}")
    print(f"  Shape: {labels.shape}")
    print(f"  Unique values: {len(np.unique(labels[labels >= 0]))}")
    print(f"  Range: [{labels[labels >= 0].min()}, {labels[labels >= 0].max()}]")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    # Example usage:
    # python gen_labels_sbox_input.py <meta_path> <output_dir>
    
    if len(sys.argv) > 1:
        meta_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "Processed"
        
        for sbox_idx in range(8):
            generate_sbox_input_labels(meta_path, sbox_idx, output_dir=output_dir)
    else:
        print("Usage: python gen_labels_sbox_input.py <meta_path> [output_dir]")
        print("Example: python gen_labels_sbox_input.py 3des-pipeline/Processed/3des/Y_meta.csv 3des-pipeline/Processed/3des")
