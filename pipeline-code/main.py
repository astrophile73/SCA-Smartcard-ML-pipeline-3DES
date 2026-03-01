import argparse
import sys
import os
import shutil
import numpy as np
from src.utils import setup_logger

logger = setup_logger("main")

def _has_trace_type(input_dir: str, file_pattern: str, card_type: str, trace_type: str) -> bool:
    try:
        from src.ingest import TraceDataset

        ds = TraceDataset(input_dir, file_pattern=file_pattern, card_type=card_type, trace_type=trace_type)
        return bool(getattr(ds, "files", []))
    except Exception:
        return False


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


def main():
    parser = argparse.ArgumentParser(description="Side-Channel Analysis Pipeline (Legacy Restoration)")
    parser.add_argument("--mode", choices=["full", "preprocess", "train", "attack"], default="full", help="Execution mode")
    # Prefer Input1 if present (repo ships datasets there); user can override.
    default_input = "Input1" if os.path.exists("Input1") else "Input"
    parser.add_argument("--input_dir", default=default_input, help="Directory containing traces")
    parser.add_argument("--output_dir", default="Output", help="Directory for final reports")
    parser.add_argument("--processed_dir", default="Processed", help="Directory for intermediate files")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (Default: 100 as per 99% accuracy mission)")
    parser.add_argument("--models_per_sbox", type=int, default=1, help="3DES models per S-Box (Default: 1)")
    parser.add_argument("--early_stop_patience", type=int, default=8, help="Early stopping patience for training (Default: 8)")
    parser.add_argument("--file_pattern", default="traces_data_*.npz", help="Pattern to match input files")
    parser.add_argument("--use_existing_pois", action="store_true", help="Use POIs from Optimization folder instead of finding new ones")
    parser.add_argument("--scan_type", choices=["all", "3des", "rsa"], default="all", help="Attack type to perform (3des, rsa, or all)")
    parser.add_argument("--card_type", choices=["visa", "mastercard", "universal"], default="universal", help="Profile (visa, mastercard, universal)")
    parser.add_argument("--opt_dir", default="Optimization", help="Directory for POIs/optimization artifacts")
    parser.add_argument("--skip_labels", action="store_true", help="Skip label generation step (use existing)")
    parser.add_argument("--start_sbox", type=int, default=1, help="Start processing from specific S-Box (1-8)")
    parser.add_argument("--model_root", default="models", help="Model root containing /3des and /rsa subfolders")
    parser.add_argument("--model_dir", help="(Deprecated) Override model root for attack/training")
    parser.add_argument("--pure_science", action="store_true", help="Scientific validation mode: Disable all reference fallbacks and report pure ML accuracy")
    parser.add_argument("--target_key", choices=["session", "master"], default="session", help="Target key for 3DES attack: Session Key (Generic) or Master Key (Derivation)")
    parser.add_argument("--enable_external_labels", action="store_true", help="Enable external 3DES key labels (e.g., Visa) from XLSX mapping")
    parser.add_argument("--label_map_xlsx", default="", help="Path to XLSX containing PROFILE/TRACK2 and 3DES key columns")
    parser.add_argument("--strict_label_mode", action="store_true", help="Fail preprocessing/training if any 3DES training row has unresolved keys")
    
    args = parser.parse_args()

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

    have_3des = _has_trace_type(args.input_dir, args.file_pattern, args.card_type, "3des")
    have_rsa = _has_trace_type(args.input_dir, args.file_pattern, args.card_type, "rsa")

    processed_3des = os.path.join(args.processed_dir, "3des")
    processed_rsa = os.path.join(args.processed_dir, "rsa")
    meta_path_3des = os.path.join(processed_3des, "Y_meta.csv")
    meta_path_rsa = os.path.join(processed_rsa, "Y_meta.csv")

    # Auto behavior: if user requests one type but only the other exists, run what exists.
    want_3des = args.scan_type in ["all", "3des"] or (args.scan_type == "rsa" and (not have_rsa) and have_3des)
    want_rsa = args.scan_type in ["all", "rsa"] or (args.scan_type == "3des" and (not have_3des) and have_rsa)

    # Migrate legacy POIs (Optimization/*.npy) into the new subfolder layout if needed.
    if have_3des:
        _maybe_migrate_legacy_pois(args.opt_dir, "3des")
    if have_rsa:
        _maybe_migrate_legacy_pois(args.opt_dir, "rsa")

    # Removed hard guard to allow recomputing POIs in attack mode.

    # STEP 1: Feature Engineering (Ingest -> Processed)
    # MANDATORY: Preprocess runs for every mode to ensure fresh data and card segregation
    if args.mode in ["full", "preprocess", "train", "attack"]:
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
                args.card_type,
                args.use_existing_pois,
                include_secrets_3des,
                args.enable_external_labels,
                args.label_map_xlsx,
                args.strict_label_mode,
            )
            if not meta_path_3des:
                meta_path_3des = os.path.join(processed_3des, "Y_meta.csv")
        else:
            logger.warning("No 3DES traces detected in input_dir.")

        if have_rsa:
            logger.info("[STEP 1] Feature Extraction (RSA traces)...")
            _, meta_path_rsa = preprocess_rsa(
                args.input_dir,
                processed_rsa,
                args.opt_dir,
                args.file_pattern,
                args.card_type,
                args.use_existing_pois,
            )
            if not meta_path_rsa:
                meta_path_rsa = os.path.join(processed_rsa, "Y_meta.csv")
        else:
            logger.warning("No RSA traces detected in input_dir.")

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
    if want_3des and have_3des and args.mode in ["full", "attack"]:
        logger.info("[STEP 3] Running 3DES Attack (Target: %s)...", args.target_key)
        try:
            from src.pipeline_3des import attack_3des

            predicted_3des, final_3des_key = attack_3des(
                processed_3des,
                model_root,
                card_type=args.card_type,
                target_key=args.target_key,
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
    if want_rsa and have_rsa and args.mode in ["full", "attack"]:
        logger.info("[STEP 4] Attacking RSA components...")
        try:
            from src.pipeline_rsa import attack_rsa

            rsa_predictions, pin_found_rsa = attack_rsa(processed_rsa, model_root, meta_path=meta_path_rsa, run_pin=True)
        except Exception as e:
            logger.warning(f"RSA attack failed: {e}")

    # [STEP 5] Final Report Generation (single merged report)
    if args.mode in ["full", "attack"]:
        logger.info("[STEP 5] Generating Reports (CSV/XLSX)...")
        from src.output_gen import OutputGenerator
        import pandas as pd

        gen = OutputGenerator(template_path="KALKi Template.csv")
        output_base = os.path.join(args.output_dir, f"Final_Report_{args.card_type}_{args.target_key}")

        frames = []
        if want_3des and have_3des and meta_path_3des and os.path.exists(meta_path_3des):
            df_3des = gen.build_rows(
                meta_path_3des,
                predicted_3des=predicted_3des,
                predicted_rsa=None,
                pin=pin_found_3des,
                final_3des_key=final_3des_key,
                pure_science=args.pure_science,
                card_type=args.card_type,
            )
            frames.append(df_3des)

        if want_rsa and have_rsa and meta_path_rsa and os.path.exists(meta_path_rsa):
            df_rsa = gen.build_rows(
                meta_path_rsa,
                predicted_3des=None,
                predicted_rsa=rsa_predictions,
                pin=pin_found_rsa,
                final_3des_key=None,
                pure_science=args.pure_science,
                card_type=args.card_type,
            )
            frames.append(df_rsa)

        if frames:
            df_all = pd.concat(frames, ignore_index=True)
            gen.write_report(df_all, output_base)
            logger.info("Pipeline Complete. Reports in: %s", args.output_dir)
        else:
            logger.warning("No report generated: no matching metadata found.")

if __name__ == "__main__":
    main()
