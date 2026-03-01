import torch
import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm

# Suppress FutureWarning for torch.load
warnings.filterwarnings('ignore', category=FutureWarning)
from src.model import get_model
from src.utils import setup_logger
from src.crypto import des_sbox_output, generate_round_keys, apply_permutation, IP, E_TABLE

logger = setup_logger("inference")

def load_model(model_path, device, input_dim=200):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found")
    model = get_model(input_dim=input_dim, num_classes=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model

def perform_attack(features_path, meta_path, model_path, sbox_idx=0, override_input=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data First to get input dim
    X = np.load(features_path).astype(np.float32)
    input_dim = X.shape[1]
    
    # Load Model
    model = load_model(model_path, device, input_dim=input_dim)
    
    df = pd.read_csv(meta_path)
    
    # For attack, we need to test all 64 key guesses (for 6 bits)
    # The Attack Algorithm (Profiled SCA):
    # 1. For each trace:
    #    - Get Model Predictions (Log Softmax / Probabilities) for output class (0..15)
    # 2. For each Key Hypothesis (0..63):
    #    - Calculate S-Box output based on Input (ATC) + KeyGuess
    #    - Look up the probability of that output from the Model
    #    - Sum log-probabilities (Log Likelihood)
    # 3. The Key Guess with max likelihood is the result.
    
    if override_input:
        logger.info(f"Starting Attack on S-Box {sbox_idx+1} using OVERRIDE inputs...")
    else:
        logger.info(f"Starting Attack on S-Box {sbox_idx+1} using {len(X)} traces...")
    
    # Pre-calculate predictions
    # Batch inference
    batch_size = 1000
    all_log_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            # Log Softmax for numerical stability
            out = torch.log_softmax(model(batch), dim=1)
            all_log_probs.append(out.cpu().numpy())
            
    all_log_probs = np.vstack(all_log_probs) # (N, 16)
    
    # Accumulate Metrics
    key_scores = np.zeros(64)
    
    # Correct Subkey (Ground Truth) for validation
    # Extract from first trace (Assuming fixed key)
    try:
        k_hex = str(df.iloc[0]['T_DES_KENC']).strip()
        if k_hex.lower() == "nan" or not k_hex:
             k_hex = "0000000000000000"
        k_hex = k_hex.ljust(16, '0')
        k_bytes = bytes.fromhex(k_hex[:16])
        
        # Parity adjustment if needed? generate_round_keys handles it?
        from src.crypto import generate_round_keys
        round_keys = generate_round_keys(k_bytes)
        rk1 = round_keys[0] # 48-bit
    except Exception as e:
        logger.warning(f"Failed Ground Truth extraction: {e}")
        rk1 = 0

    # DES S-Box mapping: SBox 1 is the first 6 bits of the 48-bit round key.
    # bits are typically indexed from 1..48 (left to right).
    # SBox 1 (idx 0) takes bits 1..6.
    # In a 48-bit integer, bit 1 is the MSB.
    # So bit 1..6 are at shift (48-6) = 42.
    # SBox 8 (idx 7) is bits 43..48, shift 0.
    shift = 42 - (sbox_idx * 6)
    correct_subkey = (rk1 >> shift) & 0x3F
    
    logger.info(f"Targeting S-Box {sbox_idx+1} | RK1 Estimate: {rk1:012x}")
    
    # Optimization: Pre-compute inputs
    # Need R0 expanded.
    
    # Only need bits relevant to SBox i
    # bits 1..6 of Expanded R0
    # Dependent on specific bits of R0.
    # Table lookup?
    
    # Prepare results container
    per_trace_guesses = []
    
    # Let's iterate traces
    for trace_idx in tqdm(range(len(df))):
        
        # 1. Reconstruct Input for this trace
        if override_input:
            # Use computed virtual input (e.g. Round 1 output for attacking Round 2)
            atc_val = override_input[trace_idx]
        else:
            # Reconstruction Logic from Metadata
            try:
                atc_bytes = []
                for j in range(8):
                    v = df.iloc[trace_idx].get(f'ATC_{j}', 0)
                    if isinstance(v, str):
                        v = int(v.replace("0x", ""), 16)
                    atc_bytes.append(int(v))
                
                # Form 64-bit int from 8 bytes (Big Endian logic: ATC_0 is MSB)
                atc_val = 0
                for j in range(8):
                    atc_val = (atc_val << 8) | atc_bytes[j]
            except Exception:
                atc_val = 0 # Default if bad metadata
        
        # IP
        ip_val = apply_permutation(atc_val, IP)
        
        # Split into Left and Right (L0, R0)
        l0 = (ip_val >> 32) & 0xFFFFFFFF
        r0 = ip_val & 0xFFFFFFFF
        
        # Expansion (r0 is 32-bit, needs width=32)
        er0 = apply_permutation(r0, E_TABLE, width=32)
        
        # Calculate scores for THIS trace only
        local_key_scores = np.zeros(64)
        
        for k_guess in range(64):
            # 6-bit subkey
            k_sbox = k_guess # We iterate 0..63
            
            # Input to SBox = E(R0) XOR K
            # We are interested in 6 bits corresponding to this SBox
            
            # Extract 6 bits for SBox idx
            # E(R0) is 48 bits. SBox 1 corresponds to bits 1..6 (indices 0..5 in 0-indexed)
            
            # Shift processing
            shift = (7 - sbox_idx) * 6
            er0_chunk = (er0 >> shift) & 0x3F
            
            val_in = er0_chunk ^ k_sbox
            
            # SBox Output
            val_out = des_sbox_output(sbox_idx, val_in)
            
            # Prob
            # all_log_probs is (N, 256)?? No, (N, 4)? No (N, 16)?
            # Wait, `all_log_probs` shape is (N, 16) -- raw Softmax output? No log_softmax?
            # It is log probabilities.
            
            prob = all_log_probs[trace_idx, val_out]
            local_key_scores[k_guess] = prob
            # Also add to Global Score
            key_scores[k_guess] += prob

        # Best guess for THIS trace
        trace_best = np.argmax(local_key_scores)
        per_trace_guesses.append(trace_best)
            
    # Rank (Global)
    sorted_idx_global = np.argsort(key_scores)[::-1] # High to Low
    rank_global = np.where(sorted_idx_global == correct_subkey)[0][0]
    best_guess_global = sorted_idx_global[0]
    
    logger.info("------------------------------------------------")
    logger.info(f"Attack Result S-Box {sbox_idx+1}")
    logger.info(f"Best Guess (Global): {hex(best_guess_global)} (Score: {key_scores[best_guess_global]:.2f})")
    
    # Show first few per-trace guesses
    logger.info(f"Per-Trace Guesses (First 5): {[hex(g) for g in per_trace_guesses[:5]]}")
    
    logger.info(f"True Key:   {hex(correct_subkey)} (Score: {key_scores[correct_subkey]:.2f})")
    logger.info(f"Final Rank (Global): {rank_global} / 64")
    logger.info("------------------------------------------------")
    
    # NEW: Prepare per-trace candidates for brute-force (Top 4 guesses for each trace)
    # We want a (N, 64) array of scores for each trace.
    # Actually, we already have `local_key_scores` in the loop but didn't save it.
    # Let's re-run or store it. Storing is better.
    
    return rank_global, sorted_idx_global, per_trace_guesses, all_log_probs

if __name__ == "__main__":
    # Test on S-Box 1 using the model trained
    rank, candidates, per_trace_guesses = perform_attack(
        "Processed/X_features.npy",
        "Processed/Y_meta.csv",
        "Optimization/best_model_opt.pth",
        sbox_idx=0
    )
