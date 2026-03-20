#!/usr/bin/env python
"""
Deep dive into the 3DES voting process to understand why recovered keys are identical.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import logging

sys.path.insert(0, r"I:\freelance\SCA Smartcard ML Pipeline-3des\pipeline-code")

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

# Import inference functions
from src.inference_3des import (
    _load_stage_features, _card_mask, _challenge_ints_from_meta, _expanded_r0,
    _load_norm, _normalize, _load_sbox_features, _load_sbox_norm,
    _ensemble_avg_probs, _score_subkey_batch, reconstruct_key_from_rk1
)

processed_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\3des"
model_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\models"

print("="*80)
print("DEBUG: 3DES Voting Process")
print("="*80)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    df0 = pd.read_csv(meta_path)
    X1 = _load_stage_features(processed_dir, stage=1)
    
    mask, detected_card_type = _card_mask(df0, "universal")
    df = df0.loc[mask].reset_index(drop=True)
    X1_masked = X1[mask]
    
    print(f"\nData loaded:")
    print(f"  Total traces: {len(df0)}")
    print(f"  Masked traces: {len(df)}")
    print(f"  Features shape: {X1_masked.shape}")
    print(f"  Card type detected: {detected_card_type}")
    
    # Get ground truth
    gt_kenc = df.iloc[0]['T_DES_KENC']
    gt_kmac = df.iloc[0]['T_DES_KMAC']
    gt_kdek = df.iloc[0]['T_DES_KDEK']
    print(f"\nGround Truth Master Keys:")
    print(f"  KENC: {gt_kenc}")
    print(f"  KMAC: {gt_kmac}")
    print(f"  KDEK: {gt_kdek}")
    
    # Compute challenges
    n_attack = 1000
    n = min(n_attack, len(X1_masked))
    challenges = _challenge_ints_from_meta(df.iloc[:n])
    er0_s1 = _expanded_r0(challenges)
    
    print(f"\nChallenge-derived values:")
    print(f"  Number of challenges: {len(challenges)}")
    print(f"  Sample ER0_S1 shape: {er0_s1.shape if hasattr(er0_s1, 'shape') else 'unknown'}")
    
    # Now run voting for KENC
    print(f"\n" + "="*80)
    print("VOTING FOR KENC")
    print("="*80)
    
    key_type = "kenc"
    norm_s1 = _load_norm(model_dir, stage=1, key_type=key_type)
    print(f"\nLoaded normalization: {norm_s1 is not None}")
    if norm_s1:
        mean, std = norm_s1
        print(f"  Mean shape: {mean.shape}, Sample: {mean[:5]}")
        print(f"  Std shape: {std.shape}, Sample: {std[:5]}")
    
    X1_norm = _normalize(X1_masked, norm_s1)
    print(f"X1 normalized shape: {X1_norm.shape}")
    print(f"X1 normalized stats: min={X1_norm.min():.4f}, max={X1_norm.max():.4f}, mean={X1_norm.mean():.4f}")
    
    sbox_votes_kenc = []
    for sbox_idx in range(1, 9):
        print(f"\n--- S-box {sbox_idx} ---")
        
        # Check for sbox-specific features
        X_sbox = _load_sbox_features(processed_dir, stage=1, sbox_idx=sbox_idx)
        if X_sbox is not None:
            X_sbox_masked = X_sbox[mask]
            sbox_norm = _load_sbox_norm(model_dir, stage=1, sbox_idx=sbox_idx, key_type=key_type)
            print(f"Sbox-specific features: shape={X_sbox_masked.shape}, normalization={'found' if sbox_norm else 'not found'}")
            X_for_inference = _normalize(X_sbox_masked, sbox_norm)
        else:
            X_for_inference = X1_norm
            print(f"Using global features: shape={X_for_inference.shape}")
        
        # Find models
        base = os.path.join(model_dir, "3des", key_type, "s1")
        model_paths = []
        if os.path.isdir(base):
            model_paths = [os.path.join(base, f) for f in sorted(os.listdir(base))
                          if f.startswith(f"sbox{sbox_idx}_model") and f.endswith(".pth")]
        
        print(f"Model paths found: {len(model_paths)}")
        if model_paths:
            for mp in model_paths:
                print(f"  {os.path.basename(mp)}")
        
        if not model_paths:
            print(f"ERROR: No models found for sbox{sbox_idx}!")
            continue
        
        # Run ensemble average
        try:
            probs = _ensemble_avg_probs(X_for_inference[:n], model_paths, device, 
                                       key_type=key_type, label_type="sbox_output")
            print(f"Probabilities shape: {probs.shape}")
            print(f"Probs stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
            
            # Score subkey
            sbox_keys = _score_subkey_batch(probs, er0_s1, sbox_idx, label_type="sbox_output")
            print(f"Sbox keys: {sbox_keys[:10]}... (showing first 10)")
            
            unique, counts = np.unique(sbox_keys, return_counts=True)
            majority_key = unique[np.argmax(counts)]
            print(f"Majority vote: {majority_key} (votes: {dict(zip(unique, counts))})")
            
            sbox_votes_kenc.append(int(majority_key))
            
        except Exception as e2:
            print(f"ERROR during scoring: {e2}")
            import traceback
            traceback.print_exc()
    
    if len(sbox_votes_kenc) == 8:
        print(f"\n" + "="*80)
        print("RECONSTRUCTING KENC FROM VOTES")
        print("="*80)
        print(f"S-box votes: {sbox_votes_kenc}")
        k1_int = reconstruct_key_from_rk1(sbox_votes_kenc)
        k1_hex = f"{k1_int:016X}"
        print(f"Reconstructed K1 (KENC): {k1_hex}")
        print(f"Ground truth KENC:        {gt_kenc}")
        print(f"Match: {k1_hex == gt_kenc}")
    else:
        print(f"\nFailed to get votes for all S-boxes: {len(sbox_votes_kenc)}/8")
        
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
