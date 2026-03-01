
import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.model import get_model
from src.utils import setup_logger
from src.crypto import des_sbox_output, generate_round_keys, apply_permutation, IP, E_TABLE
from src.pyDes import des, ECB

logger = setup_logger("inference_fixed")

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

# --- Tables for Key Reconstruction ---
_PC1 = [56, 48, 40, 32, 24, 16,  8,
      0, 57, 49, 41, 33, 25, 17,
      9,  1, 58, 50, 42, 34, 26,
     18, 10,  2, 59, 51, 43, 35,
     62, 54, 46, 38, 30, 22, 14,
      6, 61, 53, 45, 37, 29, 21,
     13,  5, 60, 52, 44, 36, 28,
     20, 12,  4, 27, 19, 11,  3]

_PC2 = [13, 16, 10, 23,  0,  4,
      2, 27, 14,  5, 20,  9,
     22, 18, 11,  3, 25,  7,
     15,  6, 26, 19, 12,  1,
     40, 51, 30, 36, 46, 54,
     29, 39, 50, 44, 32, 47,
     43, 48, 38, 55, 33, 52,
     45, 41, 49, 35, 28, 31]

# Valid for standard implementation where bit 0 is MSB?
# pyDes uses 0-based indexing.

def reconstruct_key_candidates(rk_int, round_num=1):
    """
    Reconstructs possible 64-bit Keys (56-bit effective) given a 48-bit Round Key.
    Args:
        rk_int: 48-bit integer (Round Key)
        round_num: 1-indexed round number (to reverse shifts)
    Returns:
        List of 64-bit key integers (candidates).
    """
    # 1. Reverse PC2
    # PC2 maps 56-bit C+D to 48-bit RK.
    # We create a template 56-bit array with None for missing bits.
    cd_bits = [None] * 56
    
    # Extract bits from RK
    for i in range(48):
        # Bit at index i in RK comes from PC2[i] in CD
        # RK bit i (from MSB 0)
        # Shift to extract: (47 - i)
        bit = (rk_int >> (47 - i)) & 1
        src_idx = _PC2[i]
        cd_bits[src_idx] = bit
        
    # Identify missing indices
    missing_indices = [i for i, x in enumerate(cd_bits) if x is None]
    # Should be 8 missing bits
    
    candidates = []
    
    # Iterate 2^8 possibilities
    for guess in range(256):
        # Fill missing
        current_cd = list(cd_bits)
        for i, idx in enumerate(missing_indices):
            bit = (guess >> i) & 1
            current_cd[idx] = bit
            
        # Merge C and D
        # C is first 28, D is next 28
        c = current_cd[:28]
        d = current_cd[28:]
        
        # Reverse Left Shift
        # Shifts for each round:
        # R1: 1, R2: 1, R3: 2...
        # We only implement R1 reverse for now (Shift 1)
        shift_amt = 1 # For Round 1
        
        # Rotate Right (inverse of Left Rotate)
        def ror(bits, amt):
            return bits[-amt:] + bits[:-amt]
            
        c_prev = ror(c, shift_amt)
        d_prev = ror(d, shift_amt)
        
        cd_prev = c_prev + d_prev # 56 bits
        
        # Reverse PC1
        # PC1 maps 64-bit K to 56-bit CD.
        # Create 64-bit array (ignore parity bits 7,15,23...)
        k_bits = [0] * 64
        
        for i in range(56):
            # CD bit i comes from PC1[i] in K
            src_idx = _PC1[i]
            k_bits[src_idx] = cd_prev[i]
            
        # Convert to int
        k_val = 0
        for b in k_bits:
            k_val = (k_val << 1) | b
            
        candidates.append(k_val)
        
    return candidates

def load_data(processed_dir):
    x_path = os.path.join(processed_dir, "X_features.npy")
    y_path = os.path.join(processed_dir, "Y_meta.csv")
    
    X = np.load(x_path, mmap_mode='r')
    df = pd.read_csv(y_path)
    return X, df

def load_models(opt_dir, round_name, device):
    models = []
    for sb in range(1, 9):
        model_path = os.path.join(opt_dir, f"best_model_{round_name}_sbox{sb}.pth")
        if not os.path.exists(model_path):
            # logger.warning(f"Model missing: {model_path}") # Suppress for now
            models.append(None)
            continue
            
        input_dim = 1500 
        model = get_model(input_dim=input_dim, num_classes=16).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            models.append(None)
            continue
            
        model.eval()
        models.append(model)
    return models


def precompute_model_probs(X_batch, models, device):
    """
    Runs model inference ONCE.
    """
    n_traces = X_batch.shape[0]
    bx = torch.from_numpy(X_batch).float().to(device)
    
    all_probs = []
    
    for sbox_idx, model in enumerate(models):
        if model is None:
            all_probs.append(None)
            continue
        
        with torch.no_grad():
            out = model(bx)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            probs = np.clip(probs, 1e-15, 1.0)
            all_probs.append(probs)
            
    return all_probs

def rank_candidates(all_probs, round_input_vals):
    """
    Uses precomputed probs to score candidates.
    """
    n_traces = round_input_vals.shape[0]
    scores_final = np.zeros((8, 64))
    
    for sbox_idx, probs in enumerate(all_probs):
        if probs is None: continue
        
        # Precompute E(R) for all traces
        er_vals = []
        for val in round_input_vals:
            # IP
            perm = apply_permutation(int(val), IP, 64)
            r = perm & 0xFFFFFFFF
            # E
            er = apply_permutation(r, E_TABLE, 32)
            
            # Extract 6 bits for SBox
            shift = 42 - (sbox_idx * 6)
            inp_bits = (er >> shift) & 0x3F
            er_vals.append(inp_bits)
            
        er_vals = np.array(er_vals)
        
        # Sum Scores
        for k_guess in range(64):
            sbox_in = er_vals ^ k_guess
            sbox_out = [des_sbox_output(sbox_idx, x) for x in sbox_in]
            
            # Score
            p_values = probs[np.arange(n_traces), sbox_out]
            scores_final[sbox_idx, k_guess] = np.sum(np.log(p_values))
            
    best_subkeys = np.argmax(scores_final, axis=1)
    return best_subkeys, scores_final

def run_segmented_attack(processed_dir, opt_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    if not os.path.exists(os.path.join(processed_dir, "X_features.npy")):
        logger.error("Data missing.")
        return {'K1': None, 'K2': None, 'K3': None}
        
    X_full, df = load_data(processed_dir)
    # Take slice
    N = min(len(X_full), 2000)
    X = X_full[:N]
    
    atc_inputs = []
    for idx, row in df.iloc[:N].iterrows():
        try:
            atc_bytes = bytes([int(row.get(f'ATC_{i}', 0)) for i in range(8)])
            atc_inputs.append(int.from_bytes(atc_bytes, 'big'))
        except: atc_inputs.append(0)
    atc_inputs = np.array(atc_inputs)
    
    # === ROUND 1 ===
    logger.info("=== ATTACKING ROUND 1 ===")
    models_r1 = load_models(opt_dir, "r1", device)
    
    # Precompute R1 Probs
    probs_r1 = precompute_model_probs(X, models_r1, device)
    rk1_subkeys, _ = rank_candidates(probs_r1, atc_inputs)
    
    rk1_int = 0
    for sb in range(8):
        bits = rk1_subkeys[sb] & 0x3F
        shift = 42 - (sb * 6)
        rk1_int |= (bits << shift)
        
    logger.info(f"Recovered RK1: {hex(rk1_int)}")
    
    # === RECONSTRUCT K1 ===
    k1_candidates = reconstruct_key_candidates(rk1_int, round_num=1)
    logger.info(f"Reconstructed {len(k1_candidates)} candidates for K1.")
    
    # === ROUND 2 (Decryption) ===
    logger.info("=== ATTACKING ROUND 2 ===")
    
    models_r2 = load_models(opt_dir, "r2", device)
    
    # Precompute R2 Probs ONCE
    logger.info("Precomputing Round 2 Model Probabilities...")
    probs_r2 = precompute_model_probs(X, models_r2, device)
    
    best_k1 = 0
    best_r2_score = -np.inf
    best_rk2 = 0
    
    for cand_k in tqdm(k1_candidates, desc="Checking K1 candidates"):
        # Calculate Input to R2 (Output of Enc(K1)) for all traces
        k_bytes = (int(cand_k) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'big') 
        k_des = des(k_bytes, mode=ECB)
        
        r2_inputs = []
        for val in atc_inputs:
            iv_bytes = int(val).to_bytes(8, 'big')
            enc = k_des.encrypt(iv_bytes)
            r2_inputs.append(int.from_bytes(enc, 'big'))
        r2_inputs = np.array(r2_inputs)
        
        # Attack R2 with this input assumption (Fast)
        rk2_sub, scores = rank_candidates(probs_r2, r2_inputs)
        
        # Total Score
        total_score = 0
        for sb in range(8):
            total_score += scores[sb, rk2_sub[sb]]
            
        if total_score > best_r2_score:
            best_r2_score = total_score
            best_k1 = cand_k
            best_rk2 = 0 # Construct integer
            for sb in range(8):
                bits = rk2_sub[sb] & 0x3F
                shift = 42 - (sb * 6)
                best_rk2 |= (bits << shift)
                
    logger.info(f"Verified K1: {hex(best_k1)} (Score: {best_r2_score:.2f})")
    logger.info(f"Recovered RK2: {hex(best_rk2)}")
    
    # === RECOVER K2 (from RK2) ===
    # K16 allows direct recovery of C0, D0 via PC2 inversion.
    
    # So for R2 (Decryption Round 1), the RK is K16.
    # Reverse PC2 -> C0 D0 (with missing bits).
    # Reverse PC1 -> K2.
    
    # Reconstruct K2 Candidates
    # Special function for K16 (no shift)
    def reconstruct_from_k16(rk_int):
        cd_bits = [None] * 56
        for i in range(48):
            bit = (rk_int >> (47 - i)) & 1
            src_idx = _PC2[i]
            cd_bits[src_idx] = bit
        missing_indices = [i for i, x in enumerate(cd_bits) if x is None]
        candidates = []
        for guess in range(256):
            curr_cd = list(cd_bits)
            for i, idx in enumerate(missing_indices):
                bit = (guess >> i) & 1
                curr_cd[idx] = bit
            # No shift reverse (K16 == K0/K_orig in terms of C/D)
            k_bits = [0] * 64
            for i in range(56):
                src_idx = _PC1[i]
                k_bits[src_idx] = curr_cd[i]
            k_val = 0
            for b in k_bits:
                k_val = (k_val << 1) | b
            candidates.append(k_val)
        return candidates

    k2_candidates = reconstruct_from_k16(best_rk2)
    logger.info(f"Reconstructed {len(k2_candidates)} candidates for K2.")
    
    # === ROUND 3 (Encryption) ===
    logger.info("=== ATTACKING ROUND 3 ===")
    models_r3 = load_models(opt_dir, "r3", device)
    
    # Precompute R3 Probs
    logger.info("Precomputing Round 3 Model Probabilities...")
    probs_r3 = precompute_model_probs(X, models_r3, device)
    
    best_k2 = 0
    best_r3_score = -np.inf
    best_rk3 = 0
    
    # Convert best_k1 once
    k1_des = des((int(best_k1) & 0xFFFFFFFFFFFFFFFF).to_bytes(8,'big'), mode=ECB)
    
    for cand_k in tqdm(k2_candidates, desc="Checking K2 candidates"):
        k2_des = des((int(cand_k) & 0xFFFFFFFFFFFFFFFF).to_bytes(8,'big'), mode=ECB)
        
        r3_inputs = []
        for val in atc_inputs:
            # 1. Encrypt with K1
            out1 = k1_des.encrypt(int(val).to_bytes(8,'big'))
            # 2. Decrypt with K2
            out2 = k2_des.decrypt(out1)
            r3_inputs.append(int.from_bytes(out2, 'big'))
        r3_inputs = np.array(r3_inputs)
        
        # Attack R3 (Fast)
        rk3_sub, scores = rank_candidates(probs_r3, r3_inputs)
        
        total_score = 0
        for sb in range(8):
            total_score += scores[sb, rk3_sub[sb]]
            
        if total_score > best_r3_score:
            best_r3_score = total_score
            best_k2 = cand_k
            # Construct integer
            rk_val = 0
            for sb in range(8):
                bits = rk3_sub[sb] & 0x3F
                shift = 42 - (sb * 6)
                rk_val |= (bits << shift)
            best_rk3 = rk_val
            
    logger.info(f"Verified K2: {hex(best_k2)} (Score: {best_r3_score:.2f})")
    
    # === RECOVER K3 ===
    k3_candidates = reconstruct_key_candidates(best_rk3, round_num=1)
    best_k3 = k3_candidates[0]
    
    logger.info("=== FINAL KEYS ===")
    logger.info(f"K1: {hex(best_k1)}")
    logger.info(f"K2: {hex(best_k2)}")
    logger.info(f"K3: {hex(best_k3)}")
    
    # Return keys as hex strings (16 chars)
    # Ensure unsigned
    def to_hex(val): return f"{(int(val) & 0xFFFFFFFFFFFFFFFF):016X}"
    
    return {
        'K1': to_hex(best_k1),
        'K2': to_hex(best_k2),
        'K3': to_hex(best_k3),
        'probs_r1': probs_r1,
        'rk1_subkeys': rk1_subkeys
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="Processed/Mastercard")
    parser.add_argument("--opt_dir", default="Optimization")
    args = parser.parse_args()
    
    run_segmented_attack(args.processed_dir, args.opt_dir)

if __name__ == "__main__":
    main()
