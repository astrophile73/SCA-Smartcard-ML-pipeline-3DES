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
    include_secrets=True,
    trace_type="all",
    opt_dir=None,
    external_label_map=None,
    strict_label_mode=False,
):
    """
    Extract POIs from all traces with robust alignment and Dual-Pass Global POI selection.
    """
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
        from src.gen_labels import compute_labels
        logger.info("[Pass 1] Global & Targeted POI Search...")
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
        
        from src.preprocess import align_traces
        
        for batch_traces, meta in ds.get_all_traces_iterator(batch_size=500):
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
            
            print(f"Pass 1: {n_samples} traces stats...", end='\r')
            
        denom_x_3des = np.sqrt(n_samples * sum_x2_3des - sum_x_3des**2 + 1e-10)
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
            # Fallback to variance only if correlation failed
            variance = (sum_x2_full / n_samples) - (sum_x_full / n_samples)**2
            peaks_v, _ = find_peaks(variance, distance=50, height=np.mean(variance))
            global_pois = peaks_v[np.argsort(variance[peaks_v])[::-1]][:1500]
            logger.warning("Correlation POI union failed. Falling back to Variance-based POIs.")
            global_pois_s2 = global_pois
             
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
    # REDUCED BATCH SIZE for Memory: from 2000 to 500
    for batch_traces, meta in ds.get_all_traces_iterator(batch_size=500):
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
        meta_list.append(meta[available].copy())
        
        n_processed += traces_aligned.shape[0]
        print(f"Pass 2: {n_processed} traces...", end='\r')
        
        # Memory Cleanup: Clean up after using them in the loop
        del traces_aligned
        del batch_traces
        del traces_norm

    pd.concat(meta_list, ignore_index=True).to_csv(os.path.join(output_dir, "Y_meta.csv"), index=False)
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
):
    # NOTE:
    # Historically this function auto-deleted `output_dir` when `use_existing_pois` was false.
    # That behavior is dangerous in attack mode because it can erase previously discovered POIs
    # (and any cached reference trace) and then force variance-based POIs when secrets are absent,
    # which typically destroys key-recovery accuracy. We therefore never delete output
    # directories implicitly; users can delete folders explicitly if they want a clean run.
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
    if enable_external_labels and include_secrets and trace_type == "3des":
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
    )
    
    # Generate labels for training (Critical step for Master Key alignment)
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
             
    return output_dir, os.path.join(output_dir, "Y_meta.csv")

if __name__ == "__main__":
    perform_feature_extraction("Input", "Processed", n_pois=200)
