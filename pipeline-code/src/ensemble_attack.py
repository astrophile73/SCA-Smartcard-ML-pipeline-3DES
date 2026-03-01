"""
Ensemble Attack - Average predictions from multiple models for better Rank
"""
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from src.model import get_model
from src.crypto import des_sbox_output, generate_round_keys, apply_permutation, IP, E_TABLE
from src.utils import setup_logger

logger = setup_logger("ensemble_attack")

def load_model(model_path, device, input_dim=200):
    if not os.path.exists(model_path):
        return None
    model = get_model(input_dim=input_dim, num_classes=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model

def ensemble_attack(features_path, meta_path, model_paths, sbox_idx=0):
    """
    Perform attack using ensemble of models
    
    Args:
        features_path: Path to X_sboxN.npy
        meta_path: Path to Y_meta.csv
        model_paths: List of model paths to ensemble
        sbox_idx: Which S-Box (0-7)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    X = np.load(features_path).astype(np.float32)
    meta = pd.read_csv(meta_path)
    
    logger.info(f"Ensemble Attack on S-Box {sbox_idx+1} using {len(X)} traces with {len(model_paths)} models")
    
    # Load all models
    models = []
    for mp in model_paths:
        model = load_model(mp, device, input_dim=X.shape[1])
        if model is not None:
            models.append(model)
    
    if len(models) == 0:
        logger.error("No models loaded!")
        return None, None, None
    
    logger.info(f"Loaded {len(models)} models for ensemble")
    
    # Get correct key
    from src.gen_labels import compute_labels
    labels = compute_labels(meta, sbox_idx=sbox_idx)
    correct_key = None
    
    # Extract correct key from metadata
    for idx, row in meta.iterrows():
        kenc_hex = row['T_DES_KENC']
        if pd.isna(kenc_hex) or kenc_hex == '':
            continue
        kenc_bytes = bytes.fromhex(kenc_hex)
        kenc_int = int.from_bytes(kenc_bytes[:8], 'big')
        round_keys = generate_round_keys(kenc_int)
        k1 = round_keys[0]
        shift = 42 - (sbox_idx * 6)
        correct_key = (k1 >> shift) & 0x3F
        break
    
    if correct_key is not None:
        logger.info(f"Correct Subkey for SBox {sbox_idx+1}: 0x{correct_key:x}")
    
    # Attack: Try all 64 keys
    key_scores = np.zeros(64, dtype=np.float64)
    
    for k_guess in range(64):
        score = 0.0
        
        # For each trace
        for trace_idx in tqdm(range(len(X)), desc=f"Key 0x{k_guess:02x}", leave=False):
            x_tensor = torch.from_numpy(X[trace_idx:trace_idx+1]).to(device)
            
            # Get input
            atc_bytes = []
            for i in range(8):
                atc_bytes.append(int(meta.iloc[trace_idx][f'ATC_{i}']))
            input_block = int.from_bytes(bytes(atc_bytes), 'big')
            
            # Apply IP
            input_perm = apply_permutation(input_block, IP, 64)
            R0 = input_perm & 0xFFFFFFFF
            
            # Expand R0
            expanded = apply_permutation(R0, E_TABLE, 32)
            
            # XOR with key guess
            shift = 42 - (sbox_idx * 6)
            k_full = k_guess << shift
            xor_result = expanded ^ k_full
            sb_in = (xor_result >> shift) & 0x3F
            
            # S-Box output
            val_out = des_sbox_output(sb_in, sbox_idx)
            
            # Ensemble prediction
            ensemble_prob = 0.0
            with torch.no_grad():
                for model in models:
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1)
                    ensemble_prob += probs[0, val_out].item()
            
            # Average
            ensemble_prob /= len(models)
            score += np.log(ensemble_prob + 1e-10)
        
        key_scores[k_guess] = score
    
    # Rank
    sorted_keys = np.argsort(key_scores)[::-1]
    rank = np.where(sorted_keys == correct_key)[0][0] if correct_key is not None else -1
    
    logger.info("------------------------------------------------")
    logger.info(f"Ensemble Attack Result S-Box {sbox_idx+1}")
    logger.info(f"Best Guess: 0x{sorted_keys[0]:x} (Score: {key_scores[sorted_keys[0]]:.2f})")
    if correct_key is not None:
        logger.info(f"True Key:   0x{correct_key:x} (Score: {key_scores[correct_key]:.2f})")
        logger.info(f"Final Rank: {rank} / 64")
    logger.info("------------------------------------------------")
    
    return rank, sorted_keys[0], key_scores

if __name__ == "__main__":
    # Test
    ensemble_attack(
        "Processed/X_sbox1.npy",
        "Processed/Y_meta.csv",
        ["Optimization/best_model_sbox1.pth"],
        sbox_idx=0
    )
