
import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.model import get_model
from src.utils import setup_logger
from src.inference_fixed import load_data, load_models, precompute_model_probs, rank_candidates, reconstruct_key_candidates
from src.pyDes import des, ECB

logger = setup_logger("inference_fuzzy")

def flip_bit(val, bit_idx):
    return val ^ (1 << bit_idx)

def generate_fuzzy_candidates(rk_int):
    candidates = []
    # Original
    candidates.append(rk_int)
    # Flip 1 bit (48 bits total)
    for i in range(48):
        candidates.append(flip_bit(rk_int, i))
    return candidates

def run_fuzzy_attack(processed_dir, opt_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    X_full, df = load_data(processed_dir)
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
    logger.info("=== Fuzzy Attack: R1 ===")
    models_r1 = load_models(opt_dir, "r1", device)
    probs_r1 = precompute_model_probs(X, models_r1, device)
    rk1_subkeys, _ = rank_candidates(probs_r1, atc_inputs)
    
    rk1_int = 0
    for sb in range(8):
        bits = rk1_subkeys[sb] & 0x3F
        shift = 42 - (sb * 6)
        rk1_int |= (bits << shift)
        
    logger.info(f"Base RK1: {hex(rk1_int)}")
    
    # === FUZZY GENERATION ===
    rk1_variants = generate_fuzzy_candidates(rk1_int)
    logger.info(f"Testing {len(rk1_variants)} RK1 variants (HD<=1)...")
    
    # === ROUND 2 PROBS ===
    logger.info("Precomputing R2 Probs...")
    models_r2 = load_models(opt_dir, "r2", device)
    probs_r2 = precompute_model_probs(X, models_r2, device)
    
    best_k1 = 0
    best_score = -np.inf
    
    # Iterate variants
    total_checks = 0
    
    for rk1_var in tqdm(rk1_variants, desc="Scanning RK1 Variants"):
        k1_cands = reconstruct_key_candidates(rk1_var, round_num=1)
        
        for cand_k in k1_cands:
            # Check Round 2 for this K1
            k_bytes = (int(cand_k) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'big')
            k_des = des(k_bytes, mode=ECB)
            
            # Fast Check on subset?
            # Let's do batch of 100 traces for speed
            sample_indices = np.random.choice(len(atc_inputs), 100, replace=False)
            
            r2_inputs = []
            for val in atc_inputs[sample_indices]:
                iv_bytes = int(val).to_bytes(8, 'big')
                enc = k_des.encrypt(iv_bytes)
                r2_inputs.append(int.from_bytes(enc, 'big'))
            r2_inputs = np.array(r2_inputs)
            
            # Score
            rk2_sub, scores = rank_candidates([p[sample_indices] if p is not None else None for p in probs_r2], r2_inputs)
            
            total_score = 0
            for sb in range(8):
                total_score += scores[sb, rk2_sub[sb]]
            
            if total_score > best_score:
                best_score = total_score
                best_k1 = cand_k
                
    logger.info(f"Best K1 found: {hex(best_k1)} (Score: {best_score:.2f})")
    return hex(best_k1)

if __name__ == "__main__":
    run_fuzzy_attack("Processed/Mastercard", "Optimization")
