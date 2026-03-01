
import numpy as np
import pandas as pd
import os
from src.utils import setup_logger
from src.crypto import des_sbox_output, generate_round_keys, apply_permutation, IP, E_TABLE

logger = setup_logger("gen_labels_segmented")

def generate_all_labels(processed_dir):
    y_meta_path = os.path.join(processed_dir, "Y_meta.csv")
    if not os.path.exists(y_meta_path):
        logger.error(f"{y_meta_path} not found.")
        return

    logger.info(f"Generating Multi-Round Labels from {y_meta_path}")
    df = pd.read_csv(y_meta_path)
    
    # Pre-allocate containers
    labels = {
        'R1': {i: [] for i in range(8)},
        'R2': {i: [] for i in range(8)},
        'R3': {i: [] for i in range(8)}
    }
    
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            k_hex = str(row.get('T_DES_KENC', '')).strip()
            if len(k_hex) < 16:
                 valid_indices.append(False)
                 continue
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
                
            atc_bytes = bytes([int(row.get(f'ATC_{i}', 0)) for i in range(8)])
            
            # --- R1 (E_K1) ---
            from src.pyDes import des, ECB
            des_k1 = des(k1, mode=ECB)
            k1_subkeys = generate_round_keys(k1)
            rk1 = k1_subkeys[0] 
            
            atc_int = int.from_bytes(atc_bytes, 'big')
            atc_perm = apply_permutation(atc_int, IP, 64)
            r0 = atc_perm & 0xFFFFFFFF
            er0 = apply_permutation(r0, E_TABLE, 32)
            xor_r1 = er0 ^ rk1
            
            for sb in range(8):
                shift = 42 - (sb * 6)
                inp = (xor_r1 >> shift) & 0x3F
                labels['R1'][sb].append(des_sbox_output(sb, inp))
                
            # --- R2 (D_K2) ---
            # Input is Output of R1 Invocation (Full Enc with K1)
            out_e1 = des_k1.encrypt(atc_bytes) 
            out_e1_int = int.from_bytes(out_e1, 'big')
            
            perm_e1 = apply_permutation(out_e1_int, IP, 64)
            r0_d = perm_e1 & 0xFFFFFFFF
            
            # Decryption Round 1 uses K16
            k2_subkeys = generate_round_keys(k2)
            rk_dec_first = k2_subkeys[15] 
            
            er0_d = apply_permutation(r0_d, E_TABLE, 32)
            xor_r2 = er0_d ^ rk_dec_first
            
            for sb in range(8):
                shift = 42 - (sb * 6)
                inp = (xor_r2 >> shift) & 0x3F
                labels['R2'][sb].append(des_sbox_output(sb, inp))
                
            # --- R3 (E_K3) ---
            des_k2 = des(k2, mode=ECB)
            out_d2 = des_k2.decrypt(out_e1)
            out_d2_int = int.from_bytes(out_d2, 'big')
            
            perm_d2 = apply_permutation(out_d2_int, IP, 64)
            r0_e3 = perm_d2 & 0xFFFFFFFF
            
            k3_subkeys = generate_round_keys(k3)
            rk3 = k3_subkeys[0]
            
            er0_e3 = apply_permutation(r0_e3, E_TABLE, 32)
            xor_r3 = er0_e3 ^ rk3
            
            for sb in range(8):
                shift = 42 - (sb * 6)
                inp = (xor_r3 >> shift) & 0x3F
                labels['R3'][sb].append(des_sbox_output(sb, inp))
                
            valid_indices.append(True)
            
        except Exception as e:
            valid_indices.append(False)
            
    # Save
    logger.info(f"Saving labels for {sum(valid_indices)} valid traces.")
    for r in ['R1', 'R2', 'R3']:
        for sb in range(8):
            arr = np.array(labels[r][sb])
            if len(arr) > 0:
                fname = f"Y_labels_{r.lower()}_sbox{sb+1}.npy"
                path = os.path.join(processed_dir, fname)
                np.save(path, arr)
                
    logger.info("Done.")

if __name__ == "__main__":
    generate_all_labels("Processed/Mastercard") # Default
