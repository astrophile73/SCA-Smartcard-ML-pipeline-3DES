
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from src.crypto import des_sbox_output, hamming_weight, generate_round_keys
from src.utils import setup_logger

logger = setup_logger("cpa_attack")

# Cipher Tables
# Initial Permutation (IP)
IP_TABLE = [
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7,
    56, 48, 40, 32, 24, 16, 8, 0,
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6
]
IP_TABLE = [x - 1 for x in IP_TABLE] # 0-based

# Expansion Permutation (E)
E_TABLE = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]
E_TABLE = [x - 1 for x in E_TABLE]

def run_cpa_attack():
    # 1. Load Traces and Metadata
    logger.info("Loading Traces (Mastercard)...")
    try:
        traces = np.load("Processed/Mastercard/X_features.npy")
        df = pd.read_csv("Processed/Mastercard/Y_meta.csv")
    except FileNotFoundError as e:
        logger.error(f"Cannot load data: {e}")
        return

    # Check alignment
    if len(traces) != len(df):
        logger.warning(f"Mismatch: Traces {len(traces)} vs Metadata {len(df)}")
        n = min(len(traces), len(df))
        traces = traces[:n]
        df = df.iloc[:n]

    logger.info(f"Loaded {len(traces)} traces.")

    # 2. Extract Challenges (Plaintext) from ATC columns
    pt_cols = [f'ATC_{i}' for i in range(8)]
    if not all(c in df.columns for c in pt_cols):
        logger.error("Missing ATC columns.")
        return
    plaintexts = df[pt_cols].values.astype(np.uint8) 

    # 3. Extract Session Keys from CSV
    if 'T_DES_KENC' not in df.columns:
        logger.error("Missing T_DES_KENC column.")
        return
    
    # Helper to clean key string
    def clean_key(k):
        k = str(k).strip().replace(" ", "").upper()
        # Just return first 16 chars (8 bytes) for DES R1
        return k[:16]

    keys_hex = df['T_DES_KENC'].apply(clean_key).values
    
    # 4. Compute Round Keys for every trace
    logger.info("Generating Round Keys for all traces...")
    
    rk1_chunks_all = [] # (N, 8)
    
    for k_hex in tqdm(keys_hex, desc="Generating Keys"):
        try:
            kb = bytes.fromhex(k_hex)
            # This returns list of 48-bit integers
            rks = generate_round_keys(kb) 
            rk1 = rks[0] # Round 1 key (48-bit int)
            
            # Extract 8 chunks (6 bits each)
            chunks = []
            for i in range(8):
                # S1 is top 6 bits (MSB)
                shift = 42 - (i * 6)
                val = (rk1 >> shift) & 0x3F
                chunks.append(val)
            rk1_chunks_all.append(chunks)
        except Exception as e:
            # logger.warning(f"Key error: {e}")
            rk1_chunks_all.append([0]*8)
            
    rk1_chunks_all = np.array(rk1_chunks_all, dtype=np.uint8)
    
    # 5. Compute Input to S-Boxes (Plaintext Expansion)
    logger.info("Computing S-Box Inputs...")
    pt_bits = np.unpackbits(plaintexts, axis=1) # (N, 64)
    pt_ip = pt_bits[:, IP_TABLE]
    r0 = pt_ip[:, 32:64] # Right Half (32 bits)
    r0_expanded = r0[:, E_TABLE] # (N, 48)
    
    # Pack expanded bits into 8 chunks of 6 bits
    r0_chunks = np.zeros((len(traces), 8), dtype=np.uint8)
    for sbox_idx in range(8):
        bits_slice = r0_expanded[:, sbox_idx*6 : (sbox_idx+1)*6]
        # packbits logic
        val = np.zeros(len(traces), dtype=np.uint8)
        for i in range(6):
            val = (val << 1) | bits_slice[:, i]
        r0_chunks[:, sbox_idx] = val

    # 6. CPA Correlation Loop
    logger.info("Computing Correlations...")
    
    # Standardize traces
    traces = traces - np.mean(traces, axis=0) # (N, T)
    t_sq = np.sum(traces ** 2, axis=0)
    
    results = []
    
    for sbox_idx in range(8):
        # Calculate Hypothesis: HW( SBox( R0_chunk ^ RK1_chunk ) )
        # Using vectorized true values (verification mode)
        
        # Inputs
        inp = r0_chunks[:, sbox_idx]
        key = rk1_chunks_all[:, sbox_idx]
        
        xor_val = inp ^ key
        
        # S-Box Lookup
        sbox_lut = np.array([des_sbox_output(sbox_idx, x) for x in range(64)], dtype=np.uint8)
        sbox_out = sbox_lut[xor_val]
        
        # Hamming Weight
        hw_lut = np.array([bin(x).count('1') for x in range(16)], dtype=np.uint8)
        model = hw_lut[sbox_out] # (N,)
        
        # Correlate Model vs Traces
        h_mean = np.mean(model)
        h_centered = model - h_mean
        h_sq = np.sum(h_centered ** 2)
        
        # Covariance
        cov = np.dot(h_centered, traces)
        
        # Correlation
        den = np.sqrt(t_sq * h_sq)
        den[den == 0] = 1.0
        
        corrs = np.abs(cov / den)
        
        max_corr = np.max(corrs)
        sample_idx = np.argmax(corrs)
        
        results.append({
            "SBox": int(sbox_idx + 1),
            "Max_Corr": float(max_corr),
            "Sample_Idx": int(sample_idx)
        })
        
        logger.info(f"S-Box {sbox_idx+1}: Peak Corr = {max_corr:.4f} @ {sample_idx}")

    # 7. Summary
    logger.info("=== Leakage Verification Summary ===")
    avg_corr = np.mean([r["Max_Corr"] for r in results])
    logger.info(f"Average Peak Correlation: {avg_corr:.4f}")
    
    if avg_corr > 0.05:
        logger.info("✅ SUCCESS: Leakage detected matching the Session Keys.")
    else:
        logger.warning("❌ WARNING: No significant leakage found. Check alignment, keys or power model.")

    with open("cpa_validation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_cpa_attack()
