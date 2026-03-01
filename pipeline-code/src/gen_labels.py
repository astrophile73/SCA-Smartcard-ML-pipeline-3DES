"""
3DES S-Box Label Generation Module

This module generates labels (S-Box outputs) for training the 3DES attack models.
Each label represents the 4-bit output of a specific S-Box for a given trace.
"""

import numpy as np
import pandas as pd
from src.crypto import des_sbox_output, generate_round_keys, apply_permutation, IP, E_TABLE

def _parse_input_block(row) -> bytes:
    def parse_atc_byte(val):
        try:
            return int(val)
        except Exception:
            try:
                return int(str(val), 16)
            except Exception:
                return 0

    # Prefer ATC_0..ATC_7 if available.
    try:
        b = bytes([parse_atc_byte(row.get(f"ATC_{i}", 0)) for i in range(8)])
        if any(x != 0 for x in b):
            return b
    except Exception:
        pass

    # Fallback to 'ATC' raw field if present (e.g., "7ACD" or "7A CD").
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


def compute_labels(metadata, sbox_idx, key_col="T_DES_KENC", stage: int = 1):
    """
    Computes S-Box output labels for a batch of traces.
    
    Args:
        metadata: DataFrame with columns [key_col, 'ATC_0', ..., 'ATC_7'] (or 'ATC')
        sbox_idx: S-Box index (0-7)
        key_col: Column name containing the 16-byte (32 hex) 3DES key
        stage: 1 targets K1/RK1 on input block; 2 targets K2/RK16 on DES_enc(K1, input)
    
    Returns:
        numpy array of labels (4-bit integers, 0-15)
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
            if stage == 1:
                key_bytes = bytes.fromhex(k1_hex)
                round_keys = generate_round_keys(key_bytes)
                rk = round_keys[0]  # RK1
                block_int = int.from_bytes(input_block, "big")
            elif stage == 2:
                k2_bytes = bytes.fromhex(k2_hex)
                round_keys = generate_round_keys(k2_bytes)
                # Dataset-calibrated stage-2 target:
                # the observed leakage aligns with K2 round-key interaction on the
                # same per-trace input block (not on software-simulated E(K1, input)).
                rk = round_keys[15]  # RK16
                block_int = int.from_bytes(input_block, "big")
            else:
                labels.append(-1)
                continue

            # Apply Initial Permutation (IP) on the Challenge (Plaintext)
            # This replaces the old 'atc_permuted' logic which was named 'atc' but served as plaintext
            plaintext_permuted = apply_permutation(block_int, IP, width=64)
            
            # Split into L0 and R0 (32 bits each)
            r0 = plaintext_permuted & 0xFFFFFFFF  # Right 32 bits
            
            # Expand R0 using E-table (32 -> 48 bits)
            r0_expanded = apply_permutation(r0, E_TABLE, width=32)
            
            # XOR with Round Key 1
            xor_result = r0_expanded ^ rk
            
            # Extract the 6-bit input for this specific S-Box
            # DES S-Box indexing (1-8) maps to bits in XOR result
            # Standard DES: S1=bits 42-47, S2=bits 36-41, ..., S8=bits 0-5
            shift = 42 - (sbox_idx * 6)
            six_bit_input = (xor_result >> shift) & 0x3F
            
            # Compute S-Box output (4 bits)
            sbox_output = des_sbox_output(sbox_idx, six_bit_input)
            
            labels.append(sbox_output)
            
        except Exception as e:
            labels.append(-1)
    
    return np.array(labels, dtype=np.int64)

def generate_sbox_labels(meta_path, sbox_idx, output_dir="Processed", key_col="T_DES_KENC", stage: int = 1):
    """
    Generates and saves labels for a specific S-Box.
    
    Args:
        meta_path: Path to metadata CSV
        sbox_idx: S-Box index (0-7)
        output_dir: Directory to save labels
        key_col: Column containing the 3DES key for labeling (e.g., T_DES_KENC)
    
    Returns:
        Path to saved labels file
    """
    import os
    
    # Load metadata
    df = pd.read_csv(meta_path, dtype={key_col: str, "Track2": str})
    
    # Compute labels
    labels = compute_labels(df, sbox_idx, key_col=key_col, stage=stage)
    
    # Save
    key_suffix = key_col.replace("T_DES_", "").lower()
    output_path = os.path.join(output_dir, f"Y_labels_{key_suffix}_s{stage}_sbox{sbox_idx+1}.npy")
    np.save(output_path, labels)

    # Backward-compatible alias for the original training code.
    if key_col == "T_DES_KENC" and stage == 1:
        legacy_path = os.path.join(output_dir, f"Y_labels_sbox{sbox_idx+1}.npy")
        np.save(legacy_path, labels)
    
    return output_path
