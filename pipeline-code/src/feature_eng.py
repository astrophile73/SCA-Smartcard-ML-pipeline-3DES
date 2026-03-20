import numpy as np
import pandas as pd
import os
import re
from scipy.signal import find_peaks, correlate
from tqdm import tqdm
from src.ingest import TraceDataset
from src.utils import setup_logger
from src.crypto import des_sbox_output, hamming_weight, E_TABLE, generate_round_keys, apply_permutation, IP

logger = setup_logger("feature_eng")

def get_ranked_pois(corr_vector_path, top_n=50, min_dist=50):
    if not os.path.exists(corr_vector_path):
        raise FileNotFoundError(f"{corr_vector_path} not found")
        
    corrs = np.load(corr_vector_path)
    abs_corrs = np.abs(corrs)
    peaks, _ = find_peaks(abs_corrs, distance=min_dist, height=0.04)
    
    peak_vals = abs_corrs[peaks]
    sorted_indices = np.argsort(peak_vals)[::-1]
    sorted_peaks = peaks[sorted_indices]
    
    selected_pois = sorted_peaks[:top_n]
    selected_pois = np.sort(selected_pois)
    
    logger.info(f"Selected {len(selected_pois)} POIs. Max Corr: {np.max(peak_vals) if len(peak_vals) > 0 else 0:.4f}")
    return selected_pois

def align_trace(trace, ref_trace, search_window=500):
    """
    Align trace to ref_trace using cross-correlation.
    Assumes traces are roughly aligned and only need small adjustment.
    """
    center = len(ref_trace) // 2
    if len(ref_trace) < search_window * 2:
        return trace
        
    ref_seg = ref_trace[center - search_window : center + search_window]
    trace_seg = trace[center - search_window : center + search_window]
    
    ref_seg = (ref_seg - np.mean(ref_seg)) / (np.std(ref_seg) + 1e-10)
    trace_seg = (trace_seg - np.mean(trace_seg)) / (np.std(trace_seg) + 1e-10)
    
    corr = correlate(trace_seg, ref_seg, mode='same')
    shift = np.argmax(corr) - (len(corr) // 2)
    
    if shift == 0:
        return trace
    elif shift > 0:
        return np.pad(trace[shift:], (0, shift), mode='constant')
    else:
        return np.pad(trace[:shift], (abs(shift), 0), mode='constant')

def extract_features(
    dataset_path,
    poi_indices,
    output_dir="Processed",
    file_pattern="traces_data_*.npz",
    skip_poi_search=False,
    separate_sboxes=False,
    card_type="universal",
    include_secrets=False,
    trace_type="all",
    opt_dir=None,
    external_label_map=None,
    strict_label_mode=False,
    force_variance_poi=False,
    label_type="sbox_output",
):
    """
    Extract POIs from all traces with robust alignment and Dual-Pass Global POI selection.
    
    Args:
        label_type: Type of labels to use for POI selection correlation
                   - 'sbox_output': 4-bit S-Box outputs (legacy, 0-15)
                   - 'sbox_input': 6-bit S-Box inputs (new, 0-63)
    """
    # RSA traces don't have S-box labels, so force variance-based POI selection
    if trace_type.lower() == "rsa":
        logger.info("RSA trace type detected: forcing variance-based POI selection (no S-box labels available)")
        force_variance_poi = True
    ds = TraceDataset(
        dataset_path,
        file_pattern=file_pattern,
        card_type=card_type,
        trace_type=trace_type,
        external_label_map=external_label_map,
        strict_label_mode=strict_label_mode,
    )
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if opt_dir is None:
        opt_dir = os.path.join(output_dir, "Optimization")
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
        
    feat_collections = {i: [] for i in range(8)}
    feat_collections_s2 = {i: [] for i in range(8)}
    global_feat_list_s1 = []
    global_feat_list_s2 = []
    meta_list = []
    
    T_START, T_END = 0, 300000
    ref_trace = None

    # --- PASS 1: Alignment & POI Discovery ---
    pois_per_sbox = {}
    pois_per_sbox_s2 = {}
    global_pois = None
    global_pois_s2 = None
    
    if not skip_poi_search:
        # RSA traces don't have S-box labels, so skip 3DES-specific correlation POI search
        # Force variance-based POI selection for RSA
        if trace_type == "rsa":
            logger.info("[Pass 1] RSA: Skipping correlation POI search (no S-box labels). Using variance-based POI...")
            skip_poi_search_corr = True
        else:
            skip_poi_search_corr = False
        
        if not skip_poi_search_corr:
            # Dynamically select label generator based on label_type
            # This is CRITICAL: ensures POIs are selected via correlation with the CORRECT target
            if label_type == "sbox_input":
                try:
                    from src.gen_labels_sbox_input import compute_sbox_input_labels as compute_labels
                except ImportError:
                    from gen_labels_sbox_input import compute_sbox_input_labels as compute_labels
            else:  # Default: sbox_output
                from src.gen_labels import compute_labels
            
            logger.info(f"[Pass 1] Global & Targeted POI Search (label_type={label_type})...")
            # Pure labeling uses only dataset-internal columns (no external spreadsheets).
            ref_df = None
            
            n_samples = 0
            sum_x_full, sum_x2_full = None, None
            sum_x_3des, sum_x2_3des = None, None
            sum_y_sboxes = [0.0] * 8
            sum_y2_sboxes = [0.0] * 8
            sum_xy_sboxes = [None] * 8
            n_valid_sboxes = [0] * 8
            sum_y_sboxes_s2 = [0.0] * 8
            sum_y2_sboxes_s2 = [0.0] * 8
            sum_xy_sboxes_s2 = [None] * 8
            n_valid_sboxes_s2 = [0] * 8

            # Label accumulators for all 3 key types (for saving to disk after pass)
            _KEY_TYPES_COLS = [
                ("kenc", "T_DES_KENC"),
                ("kmac", "T_DES_KMAC"),
                ("kdek", "T_DES_KDEK"),
            ]
            # all_labels[key_type][sbox_0based][stage] = list of arrays per batch
            all_labels_s1 = {kt: [[] for _ in range(8)] for kt, _ in _KEY_TYPES_COLS}
            all_labels_s2 = {kt: [[] for _ in range(8)] for kt, _ in _KEY_TYPES_COLS}
            
            from src.preprocess import align_traces
            
            for batch_traces, meta in ds.get_all_traces_iterator(batch_size=100):  # Reduced to 100 for memory safety with large files
                if ref_trace is None and len(batch_traces) > 0:
                    ref_trace = batch_traces[0].astype(np.float32)
                
                # Use fast batch alignment and float32 for memory efficiency
                # Correcting ref_trace if it was loaded as float64
                ref_float32 = ref_trace.astype(np.float32) if ref_trace is not None else None
                traces_aligned = align_traces(batch_traces.astype(np.float32), reference_trace=ref_float32)
                n = traces_aligned.shape[0]
                n_samples += n
                
                if sum_x_full is None:
                    shape = traces_aligned.shape[1]
                    sum_x_full = np.zeros(shape)
                    sum_x2_full = np.zeros(shape)
                sum_x_full += np.sum(traces_aligned, axis=0)
                sum_x2_full += np.sum(traces_aligned**2, axis=0)
                
                traces_3des = traces_aligned[:, T_START:T_END]
                if sum_x_3des is None:
                    shape_3des = traces_3des.shape[1]
                    sum_x_3des = np.zeros(shape_3des)
                    sum_x2_3des = np.zeros(shape_3des)
                    for sb in range(8):
                        sum_xy_sboxes[sb] = np.zeros(shape_3des)
                        sum_xy_sboxes_s2[sb] = np.zeros(shape_3des)
                
                sum_x_3des += np.sum(traces_3des, axis=0)
                sum_x2_3des += np.sum(traces_3des**2, axis=0)
                
                for sb in range(8):
                    # Compute KENC labels for POI correlation (default key_col)
                    labels = compute_labels(meta, sbox_idx=sb, stage=1)
                    if len(labels) != n:
                        logger.warning(
                            "Label/trace size mismatch (stage1, sbox%d): labels=%d traces=%d. Aligning with -1 padding/truncation.",
                            sb + 1,
                            len(labels),
                            n,
                        )
                        if len(labels) < n:
                            labels = np.pad(labels, (0, n - len(labels)), constant_values=-1)
                        else:
                            labels = labels[:n]
                    valid_l_mask = labels != -1
                    if np.any(valid_l_mask):
                        n_v = np.sum(valid_l_mask)
                        ys = np.array([hamming_weight(y) for y in labels[valid_l_mask]])
                        sum_y_sboxes[sb] += np.sum(ys)
                        sum_y2_sboxes[sb] += np.sum(ys**2)
                        sum_xy_sboxes[sb] += np.dot(ys, traces_3des[valid_l_mask])
                        n_valid_sboxes[sb] += n_v
                    # Store KENC labels for saving
                    all_labels_s1["kenc"][sb].append(labels.copy())

                    labels2 = compute_labels(meta, sbox_idx=sb, stage=2)
                    if len(labels2) != n:
                        logger.warning(
                            "Label/trace size mismatch (stage2, sbox%d): labels=%d traces=%d. Aligning with -1 padding/truncation.",
                            sb + 1,
                            len(labels2),
                            n,
                        )
                        if len(labels2) < n:
                            labels2 = np.pad(labels2, (0, n - len(labels2)), constant_values=-1)
                        else:
                            labels2 = labels2[:n]
                    valid_l_mask2 = labels2 != -1
                    if np.any(valid_l_mask2):
                        n_v2 = np.sum(valid_l_mask2)
                        ys2 = np.array([hamming_weight(y) for y in labels2[valid_l_mask2]])
                        sum_y_sboxes_s2[sb] += np.sum(ys2)
                        sum_y2_sboxes_s2[sb] += np.sum(ys2**2)
                        sum_xy_sboxes_s2[sb] += np.dot(ys2, traces_3des[valid_l_mask2])
                        n_valid_sboxes_s2[sb] += n_v2
                    # Store KENC stage2 labels for saving
                    all_labels_s2["kenc"][sb].append(labels2.copy())

                    # Compute KMAC and KDEK labels for saving (only if key columns present)
                    for kt, key_col in [("kmac", "T_DES_KMAC"), ("kdek", "T_DES_KDEK")]:
                        if key_col in meta.columns:
                            lbl_kt = compute_labels(meta, sbox_idx=sb, stage=1, key_col=key_col)
                            if len(lbl_kt) < n:
                                lbl_kt = np.pad(lbl_kt, (0, n - len(lbl_kt)), constant_values=-1)
                            else:
                                lbl_kt = lbl_kt[:n]
                            all_labels_s1[kt][sb].append(lbl_kt)
                            lbl_kt2 = compute_labels(meta, sbox_idx=sb, stage=2, key_col=key_col)
                            if len(lbl_kt2) < n:
                                lbl_kt2 = np.pad(lbl_kt2, (0, n - len(lbl_kt2)), constant_values=-1)
                            else:
                                lbl_kt2 = lbl_kt2[:n]
                            all_labels_s2[kt][sb].append(lbl_kt2)
                
                print(f"Pass 1: {n_samples} traces stats...", end='\r')

            # Save all accumulated labels to disk for training
            for kt, _ in _KEY_TYPES_COLS:
                for sb in range(8):
                    if all_labels_s1[kt][sb]:
                        arr = np.concatenate(all_labels_s1[kt][sb]).astype(np.int64)
                        lbl_path = os.path.join(output_dir, f"Y_labels_{kt}_s1_sbox{sb+1}.npy")
                        np.save(lbl_path, arr)
                    if all_labels_s2[kt][sb]:
                        arr2 = np.concatenate(all_labels_s2[kt][sb]).astype(np.int64)
                        lbl_path2 = os.path.join(output_dir, f"Y_labels_{kt}_s2_sbox{sb+1}.npy")
                        np.save(lbl_path2, arr2)
            logger.info("[Pass 1] Saved training labels for kenc/kmac/kdek (s1+s2) to %s", output_dir)
        else:
            # For RSA: Skip correlation POI search, just align and compute global variance POI
            logger.info("[Pass 1] Skipping correlation search for RSA (no S-box labels)...")
            from src.preprocess import align_traces
            
            n_samples = 0
            sum_x_full, sum_x2_full = None, None
            for batch_traces, meta in ds.get_all_traces_iterator(batch_size=100):  # Reduced to 100 for memory safety
                if ref_trace is None and len(batch_traces) > 0:
                    ref_trace = batch_traces[0].astype(np.float32)
                
                ref_float32 = ref_trace.astype(np.float32) if ref_trace is not None else None
                traces_aligned = align_traces(batch_traces.astype(np.float32), reference_trace=ref_float32)
                n = traces_aligned.shape[0]
                n_samples += n
                
                if sum_x_full is None:
                    shape = traces_aligned.shape[1]
                    sum_x_full = np.zeros(shape)
                    sum_x2_full = np.zeros(shape)
                sum_x_full += np.sum(traces_aligned, axis=0)
                sum_x2_full += np.sum(traces_aligned**2, axis=0)
                
                print(f"Pass 1 (RSA): {n_samples} traces stats...", end='\r')
            
            # Set dummy values for correlation variables so variance calculation works
            sum_x_3des = sum_x_full
            sum_x2_3des = sum_x2_full
            sum_y_sboxes = [0.0] * 8
            n_valid_sboxes = [0] * 8
            sum_y_sboxes_s2 = [0.0] * 8
            n_valid_sboxes_s2 = [0] * 8
        
        # DETERMINISTIC POI MODE: Skip correlation when force_variance_poi=True, always use variance-based
        # Also force variance-based POI for RSA or when no valid labels exist
        if force_variance_poi or trace_type == "rsa" or all(n == 0 for n in n_valid_sboxes):
            logger.info("Variance-based POI selection mode (deterministic for training/attack consistency)...")
            use_correlation = False
        else:
            use_correlation = include_secrets  # Use correlation only if secrets available
        
        # Safety check: if no batches were read, initialize dummy arrays for POI calculation
        if sum_x_3des is None:
            logger.warning("No trace batches loaded - initializing with dummy arrays for POI calculation")
            sum_x_3des = np.zeros(300000 - 0)  # Default T_START=0, T_END=300000
            sum_x2_3des = np.zeros(300000 - 0)
        if sum_x_full is None:
            logger.warning("Full trace stats not initialized - using dummy arrays")
            sum_x_full = np.zeros(300000)  # Default shape matching sum_x_3des
            sum_x2_full = np.zeros(300000)
        if n_samples == 0:
            logger.warning("Not enough trace samples - setting minimum of 1")
            n_samples = 1
            
        denom_x_3des = np.sqrt(n_samples * sum_x2_3des - sum_x_3des**2 + 1e-10)
        if use_correlation and not force_variance_poi:
            for sb in range(8):
                n_v = n_valid_sboxes[sb]
                if n_v < 2: continue
            denom_y = np.sqrt(n_v * sum_y2_sboxes[sb] - sum_y_sboxes[sb]**2 + 1e-10)
            numerator = n_v * sum_xy_sboxes[sb] - sum_x_3des * (sum_y_sboxes[sb] / n_v * n_samples) # Approximation for mean_x
            # Actually, mean_x is sum_x / n_samples. So numerator should be n_v * (mean_xy - mean_x * mean_y)
            # Cov(X,Y) = E[XY] - E[X]E[Y]
            # Sum(XY)/n_v - (Sum(X)/n_samples) * (Sum(Y)/n_v)
            # Multiply by n_v: Sum(XY) - (Sum(X)/n_samples) * Sum(Y)
            numerator = sum_xy_sboxes[sb] - (sum_x_3des / n_samples) * sum_y_sboxes[sb]
            correlation = np.abs(numerator / (denom_x_3des/np.sqrt(n_samples) * denom_y/np.sqrt(n_v) * n_v + 1e-10))
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            # Simpler: use Pearson formula directly
            # correlation = Cov(X,Y) / (sigma_x * sigma_y)
            # This is correct.
            # Sensitivity increase for Mastercard (Lower threshold for weak leakage with low-entropy labels)
            peaks, _ = find_peaks(correlation, distance=10, height=0.001)
            if len(peaks) > 0:
                sel_local = peaks[np.argsort(correlation[peaks])[::-1]][:800]
            else:
                sel_local = np.argsort(correlation)[-800:]
            sel_global = (sel_local + T_START).astype(int)
            pois_per_sbox[sb] = np.sort(sel_global)
            np.save(os.path.join(opt_dir, f"pois_sbox{sb+1}.npy"), pois_per_sbox[sb])
            np.save(os.path.join(opt_dir, f"pois_s1_sbox{sb+1}.npy"), pois_per_sbox[sb])

            # Stage 2 POIs (K2 / RK16)
            n_v2 = n_valid_sboxes_s2[sb]
            if n_v2 >= 2:
                denom_y2 = np.sqrt(n_v2 * sum_y2_sboxes_s2[sb] - sum_y_sboxes_s2[sb]**2 + 1e-10)
                numerator2 = sum_xy_sboxes_s2[sb] - (sum_x_3des / n_samples) * sum_y_sboxes_s2[sb]
                correlation2 = np.abs(numerator2 / (denom_x_3des/np.sqrt(n_samples) * denom_y2/np.sqrt(n_v2) * n_v2 + 1e-10))
                correlation2 = np.nan_to_num(correlation2, nan=0.0, posinf=0.0, neginf=0.0)
                peaks2, _ = find_peaks(correlation2, distance=10, height=0.001)
                if len(peaks2) > 0:
                    sel_local2 = peaks2[np.argsort(correlation2[peaks2])[::-1]][:800]
                else:
                    sel_local2 = np.argsort(correlation2)[-800:]
                sel_global2 = (sel_local2 + T_START).astype(int)
                pois_per_sbox_s2[sb] = np.sort(sel_global2)
                np.save(os.path.join(opt_dir, f"pois_s2_sbox{sb+1}.npy"), pois_per_sbox_s2[sb])
            
            # UNION POI SELECTION: Instead of variance, use the top correlation points for all S-Boxes
            all_sbox_pois = []
            all_sbox_pois_s2 = []
            for sb in range(8):
                if sb in pois_per_sbox and len(pois_per_sbox[sb]) > 0:
                    # Take top 150 from each for 1200 total (well within ZaidNet limits)
                    all_sbox_pois.append(pois_per_sbox[sb][:150])
                if sb in pois_per_sbox_s2 and len(pois_per_sbox_s2[sb]) > 0:
                    all_sbox_pois_s2.append(pois_per_sbox_s2[sb][:150])
             
            if all_sbox_pois:
                global_pois = np.unique(np.concatenate(all_sbox_pois))
                logger.info(f"Global POIs (Union of S-Boxes): {len(global_pois)} points selected.")
            if all_sbox_pois_s2:
                global_pois_s2 = np.unique(np.concatenate(all_sbox_pois_s2))
                logger.info(f"Global POIs Stage 2 (Union of S-Boxes): {len(global_pois_s2)} points selected.")
            else:
                # Fallback to variance if correlation failed - select top 200 by variance
                variance = (sum_x2_full / n_samples) - (sum_x_full / n_samples)**2
                # Select top 200 points by variance (no distance constraint)
                top_indices = np.argsort(variance)[::-1][:200]
                global_pois = np.sort(top_indices)
                logger.warning("Correlation POI union failed. Selecting %d top-variance POIs.", len(global_pois))
                global_pois_s2 = global_pois
        
        # ALWAYS USE VARIANCE when force_variance_poi=True or when correlation failed
        if force_variance_poi or global_pois is None:
            variance = (sum_x2_full / n_samples) - (sum_x_full / n_samples)**2
            # Select top 200 points by variance (no distance constraint)
            top_indices = np.argsort(variance)[::-1][:200]
            global_pois = np.sort(top_indices)
            if force_variance_poi:
                logger.info("Variance-based POI selection: %d points selected for Stage 1", len(global_pois))
            global_pois_s2 = global_pois
            
            # For pure_science mode: also create per-sbox features using same 200 POIs
            # This ensures per-sbox feature files are generated even without correlation-based selection
            if not use_correlation:
                # Assign the global 200 POIs to each S-Box (same high-variance points for all)
                for sb in range(8):
                    pois_per_sbox[sb] = global_pois
                    pois_per_sbox_s2[sb] = global_pois_s2
                    np.save(os.path.join(opt_dir, f"pois_sbox{sb+1}.npy"), pois_per_sbox[sb])
                    np.save(os.path.join(opt_dir, f"pois_s1_sbox{sb+1}.npy"), pois_per_sbox[sb])
                    np.save(os.path.join(opt_dir, f"pois_s2_sbox{sb+1}.npy"), pois_per_sbox_s2[sb])
                logger.info("Per-sbox POIs created with %d variance-based points for pure_science mode", len(global_pois))
             
        global_pois = np.sort(global_pois)
        global_pois_s2 = np.sort(global_pois_s2) if global_pois_s2 is not None else global_pois
        # Backward-compatible stage-1 global POIs
        np.save(os.path.join(opt_dir, "pois_global.npy"), global_pois)
        np.save(os.path.join(opt_dir, "pois_global_s1.npy"), global_pois)
        np.save(os.path.join(opt_dir, "pois_global_s2.npy"), global_pois_s2)
        logger.info("\nPOI Selection Complete.")

    else:
        for sb in range(8):
            p_path = os.path.join(opt_dir, f"pois_sbox{sb+1}.npy")
            pois_per_sbox[sb] = np.load(p_path) if os.path.exists(p_path) else np.array([])
            p2_path = os.path.join(opt_dir, f"pois_s2_sbox{sb+1}.npy")
            pois_per_sbox_s2[sb] = np.load(p2_path) if os.path.exists(p2_path) else np.array([])
        p_global = os.path.join(opt_dir, "pois_global_s1.npy")
        if not os.path.exists(p_global):
            p_global = os.path.join(opt_dir, "pois_global.npy")
        global_pois = np.load(p_global) if os.path.exists(p_global) else np.array([])
        p_global2 = os.path.join(opt_dir, "pois_global_s2.npy")
        global_pois_s2 = np.load(p_global2) if os.path.exists(p_global2) else global_pois

    from src.preprocess import align_traces # Import new fast alignment
    
    if ref_trace is None:
        ref_path = os.path.join(opt_dir, "reference_trace.npy")
        if os.path.exists(ref_path):
             logger.info(f"Loading reference trace from {ref_path}")
             ref_trace = np.load(ref_path)
        else:
            first_batch, _ = next(ds.get_all_traces_iterator(batch_size=1))
            if len(first_batch) > 0:
                ref_trace = first_batch[0]
                # Save it for future attacks (Green Visa)
                np.save(ref_path, ref_trace)
                logger.info(f"Saved new reference trace to {ref_path}")

    # --- PASS 2: Extraction ---
    logger.info("[Pass 2] Feature Extraction...")
    n_processed = 0
    # REDUCED BATCH SIZE for Memory: from 2000 to 100 for safe handling of large files
    for batch_traces, meta in ds.get_all_traces_iterator(batch_size=100):
        # Convert to float32 to save 50% memory immediately
        batch_traces = batch_traces.astype(np.float32)

        # Use fast batch alignment from preprocess.py
        # If ref_trace is provided, it aligns all traces to it.
        traces_aligned = align_traces(batch_traces, reference_trace=ref_trace.astype(np.float32))
        
        m = np.mean(traces_aligned, axis=1, keepdims=True)
        s = np.std(traces_aligned, axis=1, keepdims=True)
        s[s==0] = 1.0
        
        # Normalization in float32
        traces_norm = (traces_aligned - m) / s
        
        for sb in range(8):
            idx = pois_per_sbox.get(sb, np.array([]))
            if len(idx) > 0: feat_collections[sb].append(traces_norm[:, idx])
            idx2 = pois_per_sbox_s2.get(sb, np.array([]))
            if len(idx2) > 0: feat_collections_s2[sb].append(traces_norm[:, idx2])
        if global_pois is not None and len(global_pois) > 0:
            global_feat_list_s1.append(traces_norm[:, global_pois])
        if global_pois_s2 is not None and len(global_pois_s2) > 0:
            global_feat_list_s2.append(traces_norm[:, global_pois_s2])
        
        cols = ['trace_file', 'trace_idx',
                'ATC_0', 'ATC_1', 'ATC_2', 'ATC_3', 'ATC_4', 'ATC_5', 'ATC_6', 'ATC_7',
                'ATC', 'Track2', 'C7', 'IO', 'apdu', 'EncryptedPIN', 'AIP', 'IAD']
        if include_secrets:
            cols += ['T_DES_KENC', 'T_DES_KMAC', 'T_DES_KDEK']
        available = [c for c in cols if c in meta.columns]
        
        # RSA-specific: generate synthetic metadata if not available
        if trace_type.lower() == "rsa" and len(meta) == 0 and len(batch_traces) > 0:
            # Create placeholder metadata for RSA traces (no ground truth keys anyway)
            synthetic_meta = pd.DataFrame({
                'trace_file': ['synthetic'] * len(batch_traces),
                'trace_idx': list(range(n_processed, n_processed + len(batch_traces))),
                'Track2': [''] * len(batch_traces),  
                'ATC': [''] * len(batch_traces),
            })
            meta_list.append(synthetic_meta)
            logger.info(f"[RSA] Generated synthetic metadata for {len(batch_traces)} traces (n_processed={n_processed})")
        else:
            if trace_type.lower() == "rsa":
                logger.info(f"[RSA] meta.shape={meta.shape}, batch_traces.shape={batch_traces.shape}, available_cols={available}")
            meta_list.append(meta[available].copy())
        
        n_processed += traces_aligned.shape[0]
        print(f"Pass 2: {n_processed} traces...", end='\r')
        
        # Memory Cleanup: Clean up after using them in the loop
        del traces_aligned
        del batch_traces
        del traces_norm

    logger.info(f"[METADATA] meta_list has {len(meta_list)} DataFrames")
    for i, meta_df in enumerate(meta_list):
        logger.info(f"[METADATA] meta_list[{i}].shape = {meta_df.shape}")
    
    meta_combined = pd.concat(meta_list, ignore_index=True)
    logger.info(f"[METADATA] After concat: shape = {meta_combined.shape}")
    meta_combined.to_csv(os.path.join(output_dir, "Y_meta.csv"), index=False)
    if global_feat_list_s1:
        x1 = np.vstack(global_feat_list_s1)
        np.save(os.path.join(output_dir, "X_features_s1.npy"), x1)
        # Backward-compatible alias for stage 1
        np.save(os.path.join(output_dir, "X_features.npy"), x1)
    if global_feat_list_s2:
        np.save(os.path.join(output_dir, "X_features_s2.npy"), np.vstack(global_feat_list_s2))
    for sb in range(8):
        if feat_collections[sb]:
            np.save(os.path.join(output_dir, f"X_sbox{sb+1}.npy"), np.vstack(feat_collections[sb]))
        if feat_collections_s2[sb]:
            np.save(os.path.join(output_dir, f"X_s2_sbox{sb+1}.npy"), np.vstack(feat_collections_s2[sb]))
        
    logger.info(f"\nDone. Processed {n_processed} traces.")
    return None, None

def perform_feature_extraction(
    input_dir,
    output_dir,
    n_pois=200,
    file_pattern="traces_data_*.npz",
    use_existing_pois=False,
    separate_sboxes=True,
    card_type="universal",
    include_secrets=True,
    trace_type="all",
    opt_dir=None,
    enable_external_labels=False,
    label_map_xlsx=None,
    strict_label_mode=False,
    force_variance_poi=False,
    label_type="sbox_output",
    force_regenerate=False,
    cpa_label_map=None,          # dict from generate_cpa_external_label_map — skips XLSX
):
    # NOTE:
    # Historically this function auto-deleted `output_dir` when `use_existing_pois` was false.
    # That behavior is dangerous in attack mode because it can erase previously discovered POIs
    # (and any cached reference trace) and then force variance-based POIs when secrets are absent,
    # which typically destroys key-recovery accuracy. We therefore never delete output
    # directories implicitly; users can delete folders explicitly if they want a clean run.
    
    # CRITICAL FIX: Validate features match current label type
    from src.preprocess_validator import should_regenerate_features, save_preprocess_config
    
    force_regen = force_regenerate or should_regenerate_features(output_dir, label_type, verbose=True)
    if force_regen:
        logger.warning(f"[PREPROCESSING] Features will be REGENERATED (label_type={label_type})")
        use_existing_pois = False  # Force fresh POI selection
    
    if opt_dir is None:
        opt_dir = os.path.join(output_dir, "Optimization")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
                     
    skip = False
    if use_existing_pois:
        skip = True
        for sb in range(8):
             if not os.path.exists(os.path.join(opt_dir, f"pois_sbox{sb+1}.npy")): skip = False
        if not os.path.exists(os.path.join(opt_dir, "pois_global_s1.npy")) and not os.path.exists(os.path.join(opt_dir, "pois_global.npy")):
            skip = False
        if not os.path.exists(os.path.join(opt_dir, "pois_global_s2.npy")):
            skip = False
    
    external_label_map = None
    if cpa_label_map:
        # Directly-provided CPA pseudo-labels (unsupervised Visa training).
        # These take priority over an XLSX file when both are present.
        external_label_map = cpa_label_map
        logger.info("[FEAT] Using CPA pseudo-labels (%d entries).", len(cpa_label_map))
    elif enable_external_labels and include_secrets and trace_type == "3des":
        if not label_map_xlsx:
            raise ValueError("External labels enabled but --label_map_xlsx was not provided.")
        if not os.path.exists(label_map_xlsx):
            raise FileNotFoundError(f"Label map XLSX not found: {label_map_xlsx}")
        from src.external_label_map import load_external_3des_label_map

        external_label_map = load_external_3des_label_map(label_map_xlsx)
        if not external_label_map:
            raise ValueError(f"No usable 3DES label mappings loaded from: {label_map_xlsx}")
        logger.info("Loaded external 3DES label map from %s (%d entries).", label_map_xlsx, len(external_label_map))

    extract_features(
        input_dir,
        None,
        output_dir,
        file_pattern=file_pattern,
        skip_poi_search=skip,
        separate_sboxes=separate_sboxes,
        card_type=card_type,
        include_secrets=include_secrets,
        trace_type=trace_type,
        opt_dir=opt_dir,
        external_label_map=external_label_map,
        strict_label_mode=strict_label_mode,
        force_variance_poi=force_variance_poi,
        label_type=label_type,
    )
    
    # Generate labels for training (Critical step for Master Key alignment)
    # NOTE: Skip for RSA traces (they don't have 3DES keys)
    if trace_type.lower() != "rsa":
        from src.gen_labels import generate_sbox_labels
        meta_path = os.path.join(output_dir, "Y_meta.csv")
        if include_secrets and os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path, dtype=str).fillna("")
            cols = meta_df.columns.tolist()

            def _valid_key(v: str) -> bool:
                x = str(v or "").strip().replace(" ", "").upper()
                x = "".join(ch for ch in x if ch in "0123456789ABCDEF")
                return len(x) == 32 and not re.fullmatch(r"0+", x)

            required = [c for c in ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK"] if c in cols]
            if required:
                unresolved_mask = np.zeros(len(meta_df), dtype=bool)
                for c in required:
                    unresolved_mask |= ~meta_df[c].map(_valid_key).values

                unresolved_count = int(np.sum(unresolved_mask))
                if unresolved_count:
                    logger.warning(
                        "3DES metadata contains %d unresolved key rows out of %d.",
                        unresolved_count,
                        len(meta_df),
                    )
                    if strict_label_mode:
                        raise ValueError(
                            f"Strict label mode: unresolved 3DES key rows in {meta_path} "
                            f"({unresolved_count}/{len(meta_df)})."
                        )
                    # Keep only fully labeled rows for supervised training.
                    meta_df = meta_df.loc[~unresolved_mask].reset_index(drop=True)
                    meta_df.to_csv(meta_path, index=False)
                    logger.info("Filtered unresolved rows; %d labeled traces remain for training.", len(meta_df))

            for key_col in ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK"]:
                if key_col not in cols:
                    continue
                for stage in (1, 2):
                    for sb in range(8):
                        generate_sbox_labels(meta_path, sb, output_dir=output_dir, key_col=key_col, stage=stage)
    else:
        logger.info("[RSA] Skipping 3DES key validation and label generation (RSA traces don't have 3DES keys)")
        meta_path = os.path.join(output_dir, "Y_meta.csv")
    
    # Save preprocessing configuration for validation on future runs
    save_preprocess_config(output_dir, label_type)
            
    return output_dir, os.path.join(output_dir, "Y_meta.csv")

if __name__ == "__main__":
    perform_feature_extraction("Input", "Processed", n_pois=200)
