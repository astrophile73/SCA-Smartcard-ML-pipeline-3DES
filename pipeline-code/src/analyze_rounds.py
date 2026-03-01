
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from src.ingest import TraceDataset
from src.crypto import des_sbox_output, generate_round_keys, apply_permutation, IP, E_TABLE

# Standard DES P-Permutation (0-indexed)
P = [
    15, 6, 19, 20, 28, 11,
    27, 16, 0, 14, 22, 25,
    4, 17, 30, 9, 1, 7,
    23,13, 31, 26, 2, 8,
    18, 12, 29, 5, 21, 10,
    3, 24
]

# Standard DES Final Permutation (0-indexed)
FP = [
    39,  7, 47, 15, 55, 23, 63, 31,
    38,  6, 46, 14, 54, 22, 62, 30,
    37,  5, 45, 13, 53, 21, 61, 29,
    36,  4, 44, 12, 52, 20, 60, 28,
    35,  3, 43, 11, 51, 19, 59, 27,
    34,  2, 42, 10, 50, 18, 58, 26,
    33,  1, 41,  9, 49, 17, 57, 25,
    32,  0, 40,  8, 48, 16, 56, 24
]
from src.utils import setup_logger

logger = setup_logger("analyze_rounds")

def apply_des_round(block_32, rk_48):
    """Auxiliary for partial DES round logic if needed"""
    expanded = apply_permutation(block_32, E_TABLE, width=32)
    xored = expanded ^ rk_48
    # SBox outputs...
    return xored

def get_des_intermediate(pt_bytes, key_bytes, round_idx):
    """
    Compute intermediate values for analysis.
    round_idx: 1 (Enc K1), 2 (Dec K2), 3 (Enc K3)
    """
    from src.pyDes import des, ECB, PAD_NORMAL
    
    # We need access to internal state. 
    # Use our standard crypto.py functions for control or pyDes for full chain?
    # pyDes is good for full block encryption but we need SBox inputs/outputs.
    
    # Let's rebuild the chain manually to be sure.
    k1 = key_bytes[:8]
    k2 = key_bytes[8:16]
    k3 = key_bytes[16:]
    if len(key_bytes) == 16: k3 = k1
    
    # Round 1: Encrypt with K1
    # Standard DES Round 1 S-Boxes
    # We actually need what we did in gen_labels.py for R1.
    rk1 = generate_round_keys(k1)[0]
    
    # To get R2 input, we need full R1 Output? 
    # DES has 16 internal rounds. "3DES" usually means E(K1) -> D(K2) -> E(K3). E(K1) is 16 rounds.
    
    # CRITICAL CLARIFICATION:
    # "3DES is executed as THREE SEPARATE DES INVOCATIONS"
    # So "Round 1" in user's prompt likely means "The first DES invocation E(K1)".
    # "Round 2" means "The second DES invocation D(K2)".
    # "Round 3" means "The third DES invocation E(K3)".
    
    # Within E(K1), there are 16 rounds. SCA usually attacks Round 1 of the 16.
    # So for "Round 2" (D(K2)), we attack Round 1 of that Decryption process?
    # Or Round 16? Decryption usually starts with K16. 
    # But K2's "Round 1" (if inverse cipher) uses K16 of K2.
    # User's tip: "DES decryption often uses different microcode". "Attack the Output of the K2 round".
    # User also said: "K2 Input is strictly the output of Round 1 (Hardware State)"
    
    # Let's calculate the FULL output of E(K1).
    k1_des = des(k1, mode=ECB)
    r1_out = k1_des.encrypt(pt_bytes)
    
    # Now D(K2). The input is r1_out.
    # We want to attack the first round of D(K2).
    # Decryption in DES: keys are used in reverse order (K16 -> K1).
    # So the first operation in D(K2) uses subkey K16 of K2.
    
    return r1_out

def analyze_cpa_windows(input_dir, processed_dir):
    ds = TraceDataset(input_dir)
    
    # We need a batch of traces + keys
    N = 2000
    traces = []
    metadata = []
    
    # Aggregate from multiple batches until we have enough variable data
    traces = []
    metadata = []
    
    iterator = ds.get_all_traces_iterator(batch_size=N)
    for batch_t, batch_m in iterator:
        # Check variance in this batch
        # We need to look at ATC to see if it varies
        # Parse ATC first
        batch_m_processed = []
        
        has_atc_cols = 'ATC_0' in batch_m.columns
        
        current_batch_atc = []
        
        for idx, row in batch_m.iterrows():
            atc_bytes = None
            if has_atc_cols:
                 try:
                    atc_bytes = bytes([int(row.get(f'ATC_{i}', 0)) for i in range(8)])
                 except: pass
            
            if atc_bytes is None or all(b==0 for b in atc_bytes):
                # Try parsing ATC string
                atc_str = str(row.get('ATC', '')).strip()
                if atc_str and atc_str.lower() != 'nan':
                     # "7A CD" -> "00...7ACD"
                     hex_clean = atc_str.replace(" ", "").zfill(16)
                     try:
                        atc_bytes = bytes.fromhex(hex_clean)
                     except: 
                        atc_bytes = bytes([0]*8)
                else:
                     atc_bytes = bytes([0]*8)
            
            # Store the int for variance check
            atc_int = int.from_bytes(atc_bytes, 'big')
            current_batch_atc.append(atc_int)
            
            # Attach parsed bytes to row (simulated) for later use
            # We can't easily attach to row, so we'll store in a parallel list or modify DF?
            # Modifying DF row-by-row is slow.
            # Let's just keep the 'atc_bytes' in a list 
            
        current_batch_atc = np.array(current_batch_atc)
        if np.std(current_batch_atc) == 0:
            logger.warning(f"Skipping batch with constant ATC: {hex(current_batch_atc[0])}")
            continue
            
        # If variable, add to our dataset
        traces.extend(batch_t)
        # We need to store the PARSED ATC for later use to avoid re-parsing
        # So let's add 'atc_bytes' to metadata DataFrame?
        # A bit hacky. Let's just keep metadata and re-parse in the main loop (it's fast enough)
        metadata.append(batch_m)
        
        if len(traces) >= N:
            break
            
    if len(traces) == 0:
        logger.error("No traces found with variable inputs.")
        return

    traces = np.array(traces)
    metadata = pd.concat(metadata, ignore_index=True)

    n_samples = traces.shape[1]
    logger.info(f"Analyzing {len(traces)} traces with {n_samples} samples per trace")
    
    # We need known keys.
    # Assuming 'T_DES_KENC' is in metadata or we find it in ref file.
    # For Training/Analysis, we assume we can get the key.
    
    # Compute Targets
    # T1: Round 1 (E_K1, R1 of 16) - SBox Output
    # T2: Round 2 (D_K2, R1 of 16... wait, Decryption starts with K16?) 
    #     Let's handle D_K2. D is just E with reversed keys.
    #     So the "First Round of Decryption" uses Subkey 16.
    # T3: Round 3 (E_K3, R1 of 16) - SBox Output
    
    from src.pyDes import des, ECB
    from src.crypto import des_sbox_output, generate_round_keys, apply_permutation, IP, E_TABLE
    
    # Pre-compute labels
    y_r1 = []
    y_r2 = []
    y_r3 = []
    
    valid_mask = []
    
    for idx, row in metadata.iterrows():
        try:
            # Parse Key
            k_hex = str(row.get('T_DES_KENC', '')).strip()
            if len(k_hex) < 16: 
                 # Try finding in ref? simplified here.
                 valid_mask.append(False)
                 continue
            if len(k_hex) > 32: k_hex = k_hex[:32] # 16 bytes for 2-key 3DES
            
            key_bytes = bytes.fromhex(k_hex)
            if len(key_bytes) == 16:
                k1 = key_bytes[:8]
                k2 = key_bytes[8:16]
                k3 = k1
            else:
                k1 = key_bytes[:8]
                k2 = key_bytes[8:16]
                k3 = key_bytes[16:24]
                
            # Parse PT (ATC)
            atc_bytes = bytes([int(row.get(f'ATC_{i}', 0)) for i in range(8)])
            
            # --- ROUND 1 (E_K1) ---
            # Standard SBox 1 attack
            # IP
            atc_int = int.from_bytes(atc_bytes, 'big')
            atc_perm = apply_permutation(atc_int, IP, 64)
            r0 = atc_perm & 0xFFFFFFFF
            des_k1 = des(k1, mode=ECB)
            # Subkeys
            k1_subkeys = generate_round_keys(k1)
            rk1 = k1_subkeys[0] # K1 for Enc
            
            # Label R1
            # SBox 1
            er0 = apply_permutation(r0, E_TABLE, 32)
            xor = er0 ^ rk1
            shift = 42 # Sbox 1
            inp = (xor >> shift) & 0x3F
            out_r1 = des_sbox_output(0, inp)
            y_r1.append(out_r1)
            
            # --- ROUND 2 (D_K2) ---
            # Input is Output of E(K1)
            out_e1 = des_k1.encrypt(atc_bytes)
            out_e1_int = int.from_bytes(out_e1, 'big')
            
            # D(K2) starts with IP
            perm_e1 = apply_permutation(out_e1_int, IP, 64)
            r0_d = perm_e1 & 0xFFFFFFFF
            
            # Subkeys for K2 (Reversed for Decrypt?)
            # Usually Decrypt = Encrypt but keys applied K16..K1.
            # So the "First Hardware Round" of Decryption uses K16.
            k2_subkeys = generate_round_keys(k2)
            rk_dec_first = k2_subkeys[15] # K16
            
            # Label R2
            er0_d = apply_permutation(r0_d, E_TABLE, 32)
            xor_d = er0_d ^ rk_dec_first
            inp_d = (xor_d >> shift) & 0x3F
            out_r2 = des_sbox_output(0, inp_d)
            y_r2.append(out_r2)
            
            # --- ROUND 3 (E_K3) ---
            des_k2 = des(k2, mode=ECB)
            out_d2 = des_k2.decrypt(out_e1)
            # That is the input to Stage 3
            out_d2_int = int.from_bytes(out_d2, 'big')
            
            perm_d2 = apply_permutation(out_d2_int, IP, 64)
            r0_e3 = perm_d2 & 0xFFFFFFFF
            
            k3_subkeys = generate_round_keys(k3)
            rk3_first = k3_subkeys[0]
            
            # Label R3
            er0_e3 = apply_permutation(r0_e3, E_TABLE, 32)
            xor_e3 = er0_e3 ^ rk3_first
            inp_e3 = (xor_e3 >> shift) & 0x3F
            out_r3 = des_sbox_output(0, inp_e3)
            y_r3.append(out_r3)
            
            valid_mask.append(True)
            
        except Exception as e:
            # print(e)
            valid_mask.append(False)
            
    # Convert to arrays
    traces_clean = traces[valid_mask]
    y_r1 = np.array(y_r1)[valid_mask]
    y_r2 = np.array(y_r2)[valid_mask]
    y_r3 = np.array(y_r3)[valid_mask]
    
    logger.info(f"Clean Traces: {len(traces_clean)}")
    if len(traces_clean) < 10:
        logger.error("Not enough valid traces (with keys and metadata).")
        return 0,0,0
        
    logger.info(f"Label R1 Sample: {y_r1[:10]}")
    logger.info(f"Label R1 Unique: {np.unique(y_r1)}")
    
    def do_cpa(traces, labels):
        # Hamming weight
        hws = np.array([bin(x).count('1') for x in labels])
        
        # Correlate
        n = len(traces)
        x = traces.astype(np.float64)
        y = hws.astype(np.float64)
        
        # Pearson
        sum_x = np.sum(x, axis=0)
        sum_y = np.sum(y)
        sum_xy = np.dot(y, x)
        sum_x2 = np.sum(x**2, axis=0)
        sum_y2 = np.sum(y**2)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        return np.abs(numerator / (denominator + 1e-10))

    logger.info("Running CPA for Round 1...")
    cpa_r1 = do_cpa(traces_clean, y_r1)
    
    logger.info("Running CPA for Round 2...")
    cpa_r2 = do_cpa(traces_clean, y_r2)
    
    logger.info("Running CPA for Round 3...")
    cpa_r3 = do_cpa(traces_clean, y_r3)
    
    # Find Peaks
    peak_r1 = np.argmax(cpa_r1)
    peak_r2 = np.argmax(cpa_r2)
    peak_r3 = np.argmax(cpa_r3)
    
    logger.info(f"--- RESULTS ---")
    logger.info(f"Round 1 Peak: {peak_r1} (Corr: {cpa_r1[peak_r1]:.4f})")
    logger.info(f"Round 2 Peak: {peak_r2} (Corr: {cpa_r2[peak_r2]:.4f})")
    logger.info(f"Round 3 Peak: {peak_r3} (Corr: {cpa_r3[peak_r3]:.4f})")
    
    # Save for plotting if needed
    np.save("Optimization/cpa_r1.npy", cpa_r1)
    np.save("Optimization/cpa_r2.npy", cpa_r2)
    np.save("Optimization/cpa_r3.npy", cpa_r3)
    
    return peak_r1, peak_r2, peak_r3

if __name__ == "__main__":
    analyze_cpa_windows("Input", "Processed")
