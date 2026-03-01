
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from src.model_zaid import get_model
from src.utils import setup_logger
from src.crypto import apply_permutation, IP, E_TABLE, des_sbox_output

logger = setup_logger("inference_ensemble")

def load_ensemble_models(model_dir, sbox_idx, device, num_models=5):
    models = []
    # Determine input dim from saving (assuming 1500 for now, or check file)
    # We can perform a dummy load to check, or just hardcode 1500 as per training
    input_dim = 1500 
    
    for i in range(num_models):
        path = os.path.join(model_dir, f"sbox{sbox_idx}_model{i}.pth")
        if not os.path.exists(path):
            logger.warning(f"Model not found: {path}. Skipping.")
            continue
            
        model = get_model(input_dim=input_dim, num_classes=16).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        
    return models

def predict_ensemble(models, X_batch, device):
    # Returns averaged probabilities (Soft Voting)
    if not models:
        return None
        
    batch_probs = []
    with torch.no_grad():
        inputs = torch.from_numpy(X_batch).to(device)
        for model in models:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            batch_probs.append(probs.cpu().numpy())
            
    # Average probabilities across models
    avg_probs = np.mean(batch_probs, axis=0)
    return avg_probs

def run_ensemble_attack(processed_dir, model_dir, num_models=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    x_path = os.path.join(processed_dir, "X_features.npy")
    if not os.path.exists(x_path):
        logger.error(f"Features not found: {x_path}")
        return
        
    logger.info("Loading features...")
    X = np.load(x_path).astype(np.float32)
    
    # Z-Score Normalization (Critical to match training)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    
    logger.info(f"Data Loaded: {X.shape}")
    
    # We need to perform attack for each S-Box
    sbox_probs = {} # Store probabilities for each trace/sbox
    
    for sbox_idx in range(1, 9):
        logger.info(f"Running Ensemble Attack for S-Box {sbox_idx}...")
        models = load_ensemble_models(model_dir, sbox_idx, device, num_models)
        
        if not models:
            logger.error(f"No models found for S-Box {sbox_idx}")
            continue
            
        # Run prediction in batches
        batch_size = 1000
        all_probs = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            probs = predict_ensemble(models, batch, device)
            all_probs.append(probs)
            
        sbox_probs[sbox_idx] = np.vstack(all_probs)
        
    # Reconstruct Keys ( Simplified "Best Key" approach using aggregated log-likelihood would be ideal, 
    # but let's stick to the segment matching or just printing top candidates for verification first)
    
    # For meaningful testing, we really want to recover the 48-bit Round Key 1.
    # To do this properly, we need the Plaintext/ATC to invert the operations.
    # Let's load metadata.
    
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    df = pd.read_csv(meta_path, dtype={'T_DES_KENC': str})
    
    # Perform Key Recovery Attack (Simplified)
    # We will score all 64 key guesses (actually 2^6 = 64) for each S-Box based on predictions
    
    recovered_key_fragments = []
    
    for sbox_idx in range(1, 9):
        probs = sbox_probs[sbox_idx] # (N_traces, 16)
        
        # We need to test all 64 possible 6-bit subkeys for this S-Box
        key_scores = np.zeros(64)
        
        # Pre-compute inputs for speed
        # ATC (Permuted) XOR Key (Guess) = S-Box Input
        # We need permuted ATC (R0 expanded)
        
        # ... Wait, to do this correctly requires significant re-implementation of the attack logic 
        # that exists in inference_fixed.py but adapted for probabilities.
        # Given the "verified" status of `inference_fixed.py`, maybe we can just feed the *predictions* 
        # into a similar logic?
        
        # Actually, let's just save the probabilities and reuse `inference_fixed` logic if possible, 
        # or just quick implementation here:
        
        logger.info(f"Recovering key for S-Box {sbox_idx}...")
        
        # Simplified: Just find the key that maximizes the probability of the *predicted* output
        # over many traces.
        
        # 1. Recompute the "R0 Expanded" part for all traces (Same as gen_labels)
        # We can borrow this from the input dataframe if we process it again, 
        # OR we can just try to recover 3DES K1 by exhaustive search? No, too big (2^56).
        # We must recover Round Key 1 (48 bits) -> 8 chunks of 6 bits.
        
        # Let's use the Metadata to get ATC
        # We process a subset of traces (e.g. 1000) for speed
        n_attack = min(5000, len(df))
        
        for k_guess in range(64):
            score = 0
            # This loop is slow in Python, but ok for 64 guesses x 5000 traces x 8 SBoxes
            # Vectorization is better.
            
            # TODO: We need the S-Box input bits corresponding to this S-Box.
            # R0_expanded for this S-Box.
            pass

    logger.info("Ensemble Predictions Generated. To fully recover key, we need the attack logic.")
    # Saving probabilities for analysis
    # np.save(os.path.join(processed_dir, "Ensemble_Probs.npy"), sbox_probs)
    
    return sbox_probs

if __name__ == "__main__":
    preds = run_ensemble_attack("Processed/Mastercard", "Models/Ensemble_ZaidNet")
