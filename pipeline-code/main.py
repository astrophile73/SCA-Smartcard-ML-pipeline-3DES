import argparse
import sys
import os
import shutil
import numpy as np
from src.utils import setup_logger

logger = setup_logger("main")

def _pois_ready(opt_dir: str, trace_type: str) -> bool:
    """
    Attack-time guard: if correlation POI search can't run (no secrets),
    the pipeline would fall back to variance POIs and silently degrade.

    We therefore require precomputed POIs for attack runs.
    """
    if not opt_dir:
        return False

    if trace_type == "3des":
        base = os.path.join(opt_dir, "pois_3des")
        # Stage-1 + Stage-2 global POIs are required for 2-stage recovery.
        return (
            os.path.exists(os.path.join(base, "pois_global_s1.npy"))
            or os.path.exists(os.path.join(base, "pois_global.npy"))
        ) and os.path.exists(os.path.join(base, "pois_global_s2.npy"))

    if trace_type == "rsa":
        base = os.path.join(opt_dir, "pois_rsa")
        return os.path.exists(os.path.join(base, "pois_global.npy")) or os.path.exists(os.path.join(base, "pois_global_s1.npy"))

    return False


def _maybe_migrate_legacy_pois(opt_dir: str, trace_type: str) -> None:
    """
    Backward compatibility:
    Older runs stored POIs directly under `opt_dir/` (e.g., Optimization/pois_global.npy).
    New runs store them under `opt_dir/pois_3des/` and `opt_dir/pois_rsa/`.

    If legacy POIs exist and the new subfolder is missing, copy them over so attack mode
    can reuse the exact same POIs (critical for 3DES stage-2 recovery).
    """
    if not opt_dir:
        return

    if trace_type == "3des":
        legacy_global = os.path.join(opt_dir, "pois_global.npy")
        legacy_ref = os.path.join(opt_dir, "reference_trace.npy")
        new_base = os.path.join(opt_dir, "pois_3des")
        if os.path.exists(legacy_global) and not os.path.isdir(new_base):
            os.makedirs(new_base, exist_ok=True)
            for name in [
                "pois_global.npy",
                "pois_global_s1.npy",
                "pois_global_s2.npy",
                "reference_trace.npy",
                *[f"pois_sbox{i}.npy" for i in range(1, 9)],
                *[f"pois_s2_sbox{i}.npy" for i in range(1, 9)],
            ]:
                src = os.path.join(opt_dir, name)
                dst = os.path.join(new_base, name)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
            if os.path.exists(legacy_ref) and not os.path.exists(os.path.join(new_base, "reference_trace.npy")):
                try:
                    shutil.copy2(legacy_ref, os.path.join(new_base, "reference_trace.npy"))
                except Exception:
                    pass

    if trace_type == "rsa":
        legacy_global = os.path.join(opt_dir, "pois_global.npy")
        new_base = os.path.join(opt_dir, "pois_rsa")
        if os.path.exists(legacy_global) and not os.path.isdir(new_base):
            os.makedirs(new_base, exist_ok=True)
            for name in ["pois_global.npy", "pois_global_s1.npy", "reference_trace.npy"]:
                src = os.path.join(opt_dir, name)
                dst = os.path.join(new_base, name)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass


def _detect_card_family_from_traces(meta_path_3des: str) -> str:
    """
    Auto-detect the card family (visa/mastercard) from Track2 column in Y_meta.csv.
    Returns 'visa' (Track2 starts with 4), 'mastercard' (Track2 starts with 5), 
    or 'universal' if ambiguous or not found.
    """
    import pandas as pd
    
    if not os.path.exists(meta_path_3des):
        return "universal"
    
    try:
        df = pd.read_csv(meta_path_3des)
        if "Track2" not in df.columns or len(df) == 0:
            return "universal"
        
        t2_values = df["Track2"].astype(str).str.strip().str.upper()
        first_digits = t2_values.str[0].value_counts()
        
        # If 4 (Visa) is present and dominant, return 'visa'
        if first_digits.get("4", 0) > first_digits.get("5", 0):
            return "visa"
        # If 5 (Mastercard) is present and dominant, return 'mastercard'
        elif first_digits.get("5", 0) > 0:
            return "mastercard"
        else:
            return "universal"
    except Exception as e:
        logger.warning(f"Could not auto-detect card family from {meta_path_3des}: {e}")
        return "universal"


def main():
    parser = argparse.ArgumentParser(description="Side-Channel Analysis Pipeline (Legacy Restoration)")
    parser.add_argument("--mode", choices=["full", "preprocess", "train", "attack", "cpa"], default="full",
                        help="Execution mode. 'cpa' = blind statistical CPA attack: no trained models or ground-truth keys required, only trace_data + ATC columns.")
    # Prefer Input1 if present (repo ships datasets there); user can override.
    default_input = "Input1" if os.path.exists("Input1") else "Input"
    parser.add_argument("--input_dir", default=default_input, help="Directory containing traces")
    parser.add_argument("--output_dir", default="Output", help="Directory for final reports")
    parser.add_argument("--processed_dir", default="Processed", help="Directory for intermediate files")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (Default: 100 as per 99% accuracy mission)")
    parser.add_argument("--models_per_sbox", type=int, default=1, help="3DES models per S-Box (Default: 1)")
    parser.add_argument("--early_stop_patience", type=int, default=8, help="Early stopping patience for training (Default: 8)")
    parser.add_argument("--file_pattern", default="*trace*.*", help="Pattern to match input files (matches traces_data_* and trace_data_* in both .csv and .npz)")
    parser.add_argument("--use_existing_pois", action="store_true", help="Use POIs from Optimization folder instead of finding new ones")
    parser.add_argument("--scan_type", choices=["all", "3des", "rsa"], default="all", help="Attack type to perform (3des, rsa, or all)")
    parser.add_argument("--card_type", choices=["visa", "mastercard", "universal"], default="universal", help="Profile (visa, mastercard, universal)")
    parser.add_argument("--opt_dir", default="Optimization", help="Directory for POIs/optimization artifacts")
    parser.add_argument("--skip_labels", action="store_true", help="Skip label generation step (use existing)")
    parser.add_argument("--start_sbox", type=int, default=1, help="Start processing from specific S-Box (1-8)")
    parser.add_argument("--model_root", default="models", help="Model root containing /3des and /rsa subfolders")
    parser.add_argument("--model_dir", help="(Deprecated) Override model root for attack/training")
    parser.add_argument("--pure_science", action="store_true", default=True, help="Scientific validation mode:...")
    parser.add_argument("--target_key", choices=["session", "master"], default="session", help="Target key for 3DES attack: Session Key (Generic) or Master Key (Derivation)")
    parser.add_argument("--enable_external_labels", action="store_true", help="Enable external 3DES key labels (e.g., Visa) from XLSX mapping")
    parser.add_argument("--label_map_xlsx", default="", help="Path to XLSX containing PROFILE/TRACK2 and 3DES key columns")
    parser.add_argument("--strict_label_mode", action="store_true", help="Fail preprocessing/training if any 3DES training row has unresolved keys")
    parser.add_argument("--use_transfer_learning", action="store_true", help="Enable transfer learning for multi-key-type training (KENC->KMAC/KDEK)")
    parser.add_argument("--return_confidence", action="store_true", help="Include Bayesian confidence scores in attack results")
    parser.add_argument("--key_types", default="kenc,kmac,kdek", help="Comma-separated 3DES key types to train (kenc,kmac,kdek)")
    parser.add_argument("--n_attack", type=int, default=0, help="Number of traces for attack (0 = use all available)")
    parser.add_argument("--label_type", choices=["sbox_output", "sbox_input"], default="sbox_output", help="Label type for POI selection (sbox_input for new training with 6-bit inputs)")
    
    args = parser.parse_args()
    key_types = [k.strip().lower() for k in str(args.key_types).split(",") if k.strip()]

    model_root = args.model_dir if args.model_dir else args.model_root

    # Attack mode now recomputes POIs by default unless the user explicitly requests reuse.
    # This may fall back to variance-based POIs when secrets are unavailable.
    if args.mode == "attack" and not args.use_existing_pois:
        logger.warning(
            "Attack mode: recomputing POIs for this run. "
            "If secrets are unavailable, correlation POI search will fall back to variance-based POIs."
        )
    
    logger.info("==================================================")
    logger.info(f"   SCA PIPELINE (LEGACY-RESTORE) | Mode: {args.mode}")
    logger.info("==================================================")
    
    # Ensure directories
    for d in [args.processed_dir, args.output_dir, args.opt_dir, model_root]:
        if not os.path.exists(d):
            os.makedirs(d)

    have_3des = _has_trace_type(args.input_dir, args.file_pattern, "universal", "3des")
    have_rsa = _has_trace_type(args.input_dir, args.file_pattern, "universal", "rsa")

    processed_3des = os.path.join(args.processed_dir, "3des")
    processed_rsa = os.path.join(args.processed_dir, "rsa")
    meta_path_3des = os.path.join(processed_3des, "Y_meta.csv")
    meta_path_rsa = os.path.join(processed_rsa, "Y_meta.csv")

    # In attack mode, check if processed features already exist (from previous training)
    has_processed_3des = args.mode == "attack" and os.path.exists(meta_path_3des)
    has_processed_rsa = args.mode == "attack" and os.path.exists(meta_path_rsa)

    # Auto behavior: if user requests one type but only the other exists, run what exists.
    want_3des = args.scan_type in ["all", "3des"] or (args.scan_type == "rsa" and (not have_rsa) and have_3des)
    want_rsa = args.scan_type in ["all", "rsa"] or (args.scan_type == "3des" and (not have_3des) and have_rsa)

    # Migrate legacy POIs (Optimization/*.npy) into the new subfolder layout if needed.
    if have_3des or has_processed_3des:
        _maybe_migrate_legacy_pois(args.opt_dir, "3des")
    if have_rsa or has_processed_rsa:
        _maybe_migrate_legacy_pois(args.opt_dir, "rsa")

    # FIX: In attack mode, use existing processed features WITHOUT re-extraction.
    # Re-extracting features from input causes POI misalignment and breaks models.

    # STEP 1: Feature Engineering (Ingest -> Processed)
    # In attack mode with existing processed features, skip preprocessing
    if args.mode in ["full", "preprocess", "train"]:
        from src.pipeline_3des import preprocess_3des
        from src.pipeline_rsa import preprocess_rsa

        include_secrets_3des = args.mode in ["full", "train"]

        if have_3des:
            logger.info("[STEP 1] Feature Extraction (3DES traces)...")
            _, meta_path_3des = preprocess_3des(
                args.input_dir,
                processed_3des,
                args.opt_dir,
                args.file_pattern,
                "universal",  # ALWAYS extract all traces; card_type filtering happens at attack time
                args.use_existing_pois,
                include_secrets_3des,
                args.enable_external_labels,
                args.label_map_xlsx,
                args.strict_label_mode,
                force_variance_poi=(args.mode in ["full", "train", "preprocess"]),  # Use variance-based POI for training/preprocessing (stable across different feature pipelines)
                label_type=args.label_type,
                force_regenerate=(args.mode in ["full", "train"]),  # Force fresh features for training
            )
            if not meta_path_3des:
                meta_path_3des = os.path.join(processed_3des, "Y_meta.csv")
        elif args.mode != "attack":
            logger.warning("No 3DES traces detected in input_dir.")

        if have_rsa:
            logger.info("[STEP 1] Feature Extraction (RSA traces)...")
            include_secrets_rsa = args.mode in ["full", "train"]
            _, meta_path_rsa = preprocess_rsa(
                args.input_dir,
                processed_rsa,
                args.opt_dir,
                args.file_pattern,
                "universal",  # ALWAYS extract all traces; card_type filtering happens at attack time
                args.use_existing_pois,
                include_secrets_rsa,
                args.enable_external_labels,
                args.label_map_xlsx,
                args.strict_label_mode,
                force_variance_poi=(args.mode in ["full", "train", "preprocess"]),
                label_type=args.label_type,
                force_regenerate=(args.mode in ["full", "train"]),
            )
            if not meta_path_rsa:
                meta_path_rsa = os.path.join(processed_rsa, "Y_meta.csv")
        elif args.mode != "attack":
            logger.warning("No RSA traces detected in input_dir.")
    elif args.mode == "attack":
        logger.info("[STEP 1] Attack mode: Using existing processed features...")

    # [STEP 2] Training Models (3DES & RSA)
    if args.mode in ["full", "train"]:
        logger.info("[STEP 2] Starting Training (3DES & RSA)...")
        from src.pipeline_3des import train_3des
        from src.pipeline_rsa import train_rsa

        if want_3des and have_3des:
            train_3des(
                processed_3des,
                model_root,
                args.models_per_sbox,
                args.epochs,
                args.early_stop_patience,
                use_transfer_learning=args.use_transfer_learning,
                key_types=key_types,
                label_type=args.label_type,
            )
        elif want_3des and not have_3des:
            logger.warning("Skipping 3DES training: no 3DES traces detected.")

        if want_rsa and have_rsa:
            trained = train_rsa(processed_rsa, model_root, epochs=args.epochs, batch_size=32)
            if not trained:
                logger.warning("RSA training skipped: dataset-internal RSA labels not found.")

    pin_found_3des = None
    pin_found_rsa = None
    final_3des_key = None
    predicted_3des = None
    rsa_predictions = None

    # [STEP 3] 3DES Attack
    if args.mode == "cpa" and want_3des:
        # ----------------------------------------------------------------
        # [STEP 3-CPA] Blind Pearson CPA — no models or ground-truth keys
        # ----------------------------------------------------------------
        logger.info("[STEP 3-CPA] Blind CPA key recovery (no ML models required)...")
        try:
            from src.cpa_attack import run_cpa_blind_recovery

            cpa_out_dir = os.path.join(args.output_dir, "cpa")
            
            # Auto-detect card family if universal
            detected_card_type = args.card_type
            if args.card_type == "universal":
                meta_path_3des = os.path.join(processed_3des, "Y_meta.csv")
                detected_card_type = _detect_card_family_from_traces(meta_path_3des)
                if detected_card_type == "universal":
                    # If still ambiguous, default to visa
                    detected_card_type = "visa"
                else:
                    logger.info(f"[AUTO-DETECT] Card family detected: {detected_card_type}")
            
            predicted_3des = run_cpa_blind_recovery(
                input_dir=args.input_dir,
                processed_dir=processed_3des,
                output_dir=cpa_out_dir,
                card_type=detected_card_type,
                n_traces_max=args.n_attack,
            )
            meta_path_3des   = os.path.join(processed_3des, "Y_meta.csv")
            has_processed_3des = True
            if predicted_3des:
                logger.info(
                    "[CPA] Session key recovered: K1=%s | K2=%s",
                    predicted_3des.get("cpa_k1", "?"),
                    predicted_3des.get("cpa_k2", "?"),
                )
        except Exception as e:
            logger.error("CPA attack failed: %s", e)
            predicted_3des = None

    elif (want_3des and (have_3des or has_processed_3des)) and args.mode in ["full", "attack"]:
        logger.info("[STEP 3] Running 3DES Attack (Target: %s)...", args.target_key)
        try:
            from src.pipeline_3des import attack_3des
            
            # Auto-detect card family if universal
            detected_card_type = args.card_type
            if args.card_type == "universal":
                meta_path_3des = os.path.join(processed_3des, "Y_meta.csv")
                detected_card_type = _detect_card_family_from_traces(meta_path_3des)
                if detected_card_type != "universal":
                    logger.info(f"[AUTO-DETECT] Card family detected: {detected_card_type}")

            predicted_3des, final_3des_key = attack_3des(
                processed_3des,
                model_root,
                card_type=detected_card_type,
                target_key=args.target_key,
                return_confidence=args.return_confidence,
                n_attack=args.n_attack,
                pure_science=args.pure_science,
            )
            if predicted_3des:
                logger.info("Recovered 3DES keys (pure ML): %s", list(predicted_3des.keys()))
                missing = [k for k in ["3DES_KENC", "3DES_KMAC", "3DES_KDEK"] if k not in predicted_3des]
                if missing:
                    logger.warning(
                        "3DES recovery incomplete (missing %s). This usually means the required models "
                        "(and/or stage-2 models) are not present under %s.",
                        ",".join(missing),
                        model_root,
                    )
            elif final_3des_key:
                logger.info("Recovered Master Key-derived 3DES key.")
            else:
                logger.warning("No 3DES keys recovered. Check that 3DES models exist under %s.", model_root)
        except Exception as e:
            logger.error(f"3DES Attack Failed: {e}")
            predicted_3des = None

    # [STEP 4] RSA Attack
    if (want_rsa and (have_rsa or has_processed_rsa)) and args.mode in ["full", "attack"]:
        logger.info("[STEP 4] Attacking RSA components...")
        try:
            from src.pipeline_rsa import attack_rsa

            rsa_predictions, pin_found_rsa = attack_rsa(processed_rsa, model_root, meta_path=meta_path_rsa, run_pin=True)
        except Exception as e:
            logger.warning(f"RSA attack failed: {e}")

    # [STEP 5] Final Report Generation (unified single rows with both 3DES and RSA)
    if args.mode in ["full", "attack", "cpa"]:
        logger.info("[STEP 5] Generating Reports (CSV/XLSX)...")
        from src.output_gen import OutputGenerator, _normalize_hex
        import pandas as pd

        gen = OutputGenerator(template_path="KALKi Template.csv")
        output_base = os.path.join(args.output_dir, f"Final_Report_{args.card_type}_{args.target_key}")

        # When both 3DES and RSA exist, only generate rows for traces that have BOTH results
        if want_3des and (have_3des or has_processed_3des) and want_rsa and (have_rsa or has_processed_rsa) and meta_path_3des and meta_path_rsa:
            # Read both metadata files
            df_3des_meta = pd.read_csv(meta_path_3des)
            df_rsa_meta = pd.read_csv(meta_path_rsa)
            
            # Determine the number of traces that have BOTH results
            n_3des_traces = len(df_3des_meta)
            n_rsa_traces = len(df_rsa_meta)
            n_both_traces = min(n_3des_traces, n_rsa_traces)
            
            # If either dataset is empty, fall back to single-attack report
            if n_both_traces == 0:
                logger.warning(f"Unified report skipped: 3DES has {n_3des_traces} rows, RSA has {n_rsa_traces} rows. Falling back to single-attack.")
                if n_3des_traces > 0:
                    df_3des = gen.build_rows(
                        meta_path_3des,
                        predicted_3des=predicted_3des,
                        predicted_rsa=None,
                        pin=pin_found_3des,
                        final_3des_key=final_3des_key,
                        pure_science=args.pure_science,
                        card_type=args.card_type,
                    )
                    gen.write_report(df_3des, output_base)
                    logger.info("Pipeline Complete. Generated 3DES-only report with %d rows (RSA data unavailable)", len(df_3des))
                elif n_rsa_traces > 0:
                    df_rsa = gen.build_rows(
                        meta_path_rsa,
                        predicted_3des=None,
                        predicted_rsa=rsa_predictions,
                        pin=pin_found_rsa,
                        final_3des_key=None,
                        pure_science=args.pure_science,
                        card_type=args.card_type,
                    )
                    gen.write_report(df_rsa, output_base)
                    logger.info("Pipeline Complete. Generated RSA-only report with %d rows (3DES data unavailable)", len(df_rsa))
                else:
                    logger.error("No data available in either 3DES or RSA metadata files.")
            else:
                logger.info(f"Generating unified report: {n_3des_traces} 3DES traces, {n_rsa_traces} RSA traces, combining {n_both_traces}")
                
                # Limit 3DES metadata to rows that have RSA equivalents
                df_3des_limited = df_3des_meta.iloc[:n_both_traces].reset_index(drop=True)
                
                # Generate unified rows with BOTH predictions
                df_unified = gen.build_rows(
                    meta_path_3des,  # Use path for validation, but we'll pass limited metadata below
                    predicted_3des=predicted_3des,
                    predicted_rsa=rsa_predictions,
                    pin=pin_found_3des if want_3des else pin_found_rsa,
                    final_3des_key=final_3des_key,
                    pure_science=args.pure_science,
                    card_type=args.card_type,
                )
                
                # Keep only rows that have both predictions (first n_both_traces)
                if len(df_unified) > n_both_traces:
                    df_unified = df_unified.iloc[:n_both_traces].reset_index(drop=True)
                
                gen.write_report(df_unified, output_base)
                logger.info("Pipeline Complete. Generated unified report with %d rows (both 3DES + RSA)", len(df_unified))
        
        # Fall back to single attack if only 3DES or RSA available
        elif want_3des and (have_3des or has_processed_3des) and meta_path_3des:
            df_3des = gen.build_rows(
                meta_path_3des,
                predicted_3des=predicted_3des,
                predicted_rsa=None,
                pin=pin_found_3des,
                final_3des_key=final_3des_key,
                pure_science=args.pure_science,
                card_type=args.card_type,
            )
            gen.write_report(df_3des, output_base)
            logger.info("Pipeline Complete. Generated 3DES-only report with %d rows", len(df_3des))
            
        elif want_rsa and (have_rsa or has_processed_rsa) and meta_path_rsa:
            df_rsa = gen.build_rows(
                meta_path_rsa,
                predicted_3des=None,
                predicted_rsa=rsa_predictions,
                pin=pin_found_rsa,
                final_3des_key=None,
                pure_science=args.pure_science,
                card_type=args.card_type,
            )
            gen.write_report(df_rsa, output_base)
            logger.info("Pipeline Complete. Generated RSA-only report with %d rows", len(df_rsa))
        
        else:
            logger.warning("No report generated: no matching metadata found.")

if __name__ == "__main__":
    main()
