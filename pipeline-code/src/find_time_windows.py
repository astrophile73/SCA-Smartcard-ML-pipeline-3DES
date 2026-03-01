
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.utils import setup_logger
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

logger = setup_logger("find_time_windows")

def find_windows():
    processed_dir = "Processed/Visa"
    x_path = os.path.join(processed_dir, "X_features.npy")
    y_path = os.path.join(processed_dir, "Y_meta.csv")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        logger.error("Files missing in Processed/Mastercard")
        return

    logger.info(f"Loading X from {x_path}")
    # Use mmap or load subset to save memory if needed
    # X usually [N, Time]
    X = np.load(x_path, mmap_mode='r')
    N_total = X.shape[0]
    # We only need ~2000 traces for CPA
    N = min(2000, N_total)
    
    X_slice = np.array(X[:N], dtype=np.float64)
    logger.info(f"Loaded X slice: {X_slice.shape}")
    
    logger.info(f"Loading Metadata from {y_path}")
    df = pd.read_csv(y_path)
    df_slice = df.iloc[:N]
    
    # Compute Labels
    labels_r1 = []
    labels_r2 = []
    labels_r3 = []
    
    valid_mask = []
    
    for idx, row in df_slice.iterrows():
        try:
            # Parse Key
            # Y_meta.csv uses 'T_DES_KENC'
            k_hex = str(row.get('T_DES_KENC', '')).strip()
            if len(k_hex) < 16:
                 valid_mask.append(False)
                 continue
            # Handle potential long key
            if len(k_hex) > 32: k_hex = k_hex[:32]
            
            key_bytes = bytes.fromhex(k_hex)
            if len(key_bytes) == 16:
                k1 = key_bytes[:8]
                k2 = key_bytes[8:16]
                k3 = k1
            else:
                k1 = key_bytes[:8]
                k2 = key_bytes[8:16]
                k3 = key_bytes[16:24]
                
            # Parse ATC (Use ATC_0...ATC_7 columns which look reliable)
            atc_bytes = bytes([int(row.get(f'ATC_{i}', 0)) for i in range(8)])
            
            # --- Logic Copied from Analyze Rounds ---
            from src.pyDes import des, ECB
            
            # ROUND 1 (E_K1) - SBox 1
            atc_int = int.from_bytes(atc_bytes, 'big')
            atc_perm = apply_permutation(atc_int, IP, 64)
            r0 = atc_perm & 0xFFFFFFFF
            des_k1 = des(k1, mode=ECB)
            k1_subkeys = generate_round_keys(k1)
            rk1 = k1_subkeys[0] 
            er0 = apply_permutation(r0, E_TABLE, 32)
            xor = er0 ^ rk1
            shift = 42
            inp = (xor >> shift) & 0x3F
            out_r1 = des_sbox_output(0, inp)
            labels_r1.append(out_r1)
            
            # ROUND 2 (D_K2) - SBox 1
            out_e1 = des_k1.encrypt(atc_bytes)
            out_e1_int = int.from_bytes(out_e1, 'big')
            
            perm_e1 = apply_permutation(out_e1_int, IP, 64)
            r0_d = perm_e1 & 0xFFFFFFFF
            
            # Subkeys for K2 (Decryption Round 1 uses K16)
            k2_subkeys = generate_round_keys(k2)
            rk_dec_first = k2_subkeys[15] # K16
            
            er0_d = apply_permutation(r0_d, E_TABLE, 32)
            xor_d = er0_d ^ rk_dec_first
            inp_d = (xor_d >> shift) & 0x3F
            out_r2_val = des_sbox_output(0, inp_d)
            labels_r2.append(out_r2_val)
            
            # ROUND 3 (E_K3) - SBox 1
            des_k2 = des(k2, mode=ECB)
            out_d2 = des_k2.decrypt(out_e1)
            out_d2_int = int.from_bytes(out_d2, 'big')
            
            perm_d2 = apply_permutation(out_d2_int, IP, 64)
            r0_e3 = perm_d2 & 0xFFFFFFFF
            
            k3_subkeys = generate_round_keys(k3)
            rk3_first = k3_subkeys[0]
            
            er0_e3 = apply_permutation(r0_e3, E_TABLE, 32)
            xor_e3 = er0_e3 ^ rk3_first
            inp_e3 = (xor_e3 >> shift) & 0x3F
            out_r3_val = des_sbox_output(0, inp_e3)
            labels_r3.append(out_r3_val)
            valid_mask.append(True)
            
        except Exception as e:
            # print(e)
            valid_mask.append(False)
            
    # CPA
    y_r1 = np.array(labels_r1)
    y_r2 = np.array(labels_r2)
    y_r3 = np.array(labels_r3)
    
    logger.info(f"Valid Traces: {len(y_r1)}")
    logger.info(f"Unique Labels R1: {len(np.unique(y_r1))}")
    logger.info(f"Unique Labels R2: {len(np.unique(y_r2))}")
    logger.info(f"Unique Labels R3: {len(np.unique(y_r3))}")
    
    def do_cpa(traces, labels):
        hws = np.array([bin(x).count('1') for x in labels]).astype(np.float64)
        n = len(traces)
        x = traces
        y = hws
        sum_x = np.sum(x, axis=0)
        sum_y = np.sum(y)
        sum_xy = np.dot(y, x)
        sum_x2 = np.sum(x**2, axis=0)
        sum_y2 = np.sum(y**2)
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        return np.abs(numerator / (denominator + 1e-10))

    logger.info("Computing CPA R1...")
    cpa_r1 = do_cpa(X_slice[valid_mask], y_r1)
    peak_r1 = np.argmax(cpa_r1)
    
    logger.info("Computing CPA R2...")
    cpa_r2 = do_cpa(X_slice[valid_mask], y_r2)
    peak_r2 = np.argmax(cpa_r2)
    
    logger.info("Computing CPA R3...")
    cpa_r3 = do_cpa(X_slice[valid_mask], y_r3)
    peak_r3 = np.argmax(cpa_r3)
    
    logger.info("--- PEAK LOCATIONS ---")
    logger.info(f"R1 Peak: {peak_r1} (Corr: {cpa_r1[peak_r1]:.4f})")
    logger.info(f"R2 Peak: {peak_r2} (Corr: {cpa_r2[peak_r2]:.4f})")
    logger.info(f"R3 Peak: {peak_r3} (Corr: {cpa_r3[peak_r3]:.4f})")
    
    # Heuristic Window Size
    # Usually Round 1 is X samples wide.
    # We define offset relative to R1.
    return peak_r1, peak_r2, peak_r3

if __name__ == "__main__":
    find_windows()
