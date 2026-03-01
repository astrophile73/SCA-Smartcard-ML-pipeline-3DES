import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.feature_eng import perform_feature_extraction
from src.gen_labels_rsa import generate_rsa_labels
from src.inference_rsa import attack_all_rsa_components
from src.pin_extraction import extract_pin_from_trace_rsa
from src.train_rsa import train_rsa_model
from src.utils import setup_logger

logger = setup_logger("pipeline_rsa")


def preprocess_rsa(
    input_dir: str,
    processed_dir: str,
    opt_dir: str,
    file_pattern: str,
    card_type: str,
    use_existing_pois: bool,
) -> Tuple[Optional[str], Optional[str]]:
    return perform_feature_extraction(
        input_dir,
        processed_dir,
        n_pois=1500,
        file_pattern=file_pattern,
        use_existing_pois=use_existing_pois,
        card_type=card_type,
        include_secrets=False,
        trace_type="rsa",
        opt_dir=os.path.join(opt_dir, "pois_rsa"),
    )


def _has_rsa_labels(meta_path: str) -> bool:
    try:
        df = pd.read_csv(meta_path, nrows=500)
    except Exception:
        return False
    label_cols = [c for c in df.columns if c.startswith("RSA_CRT_")]
    if not label_cols:
        return False
    # Require at least one non-empty entry
    sub = df[label_cols].astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return not sub.dropna(how="all").empty


def train_rsa(
    processed_dir: str,
    model_root: str,
    epochs: int = 100,
    batch_size: int = 32,
) -> bool:
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    if not os.path.exists(meta_path):
        logger.warning("RSA training skipped: missing %s", meta_path)
        return False

    if not _has_rsa_labels(meta_path):
        logger.warning("RSA training skipped: no RSA_CRT_* labels found in metadata.")
        return False

    meta_df = pd.read_csv(meta_path)
    label_paths = generate_rsa_labels(meta_df, None, processed_dir)

    model_dir = os.path.join(model_root, "rsa")
    os.makedirs(model_dir, exist_ok=True)

    suffix_map = {
        "RSA_CRT_P": "p",
        "RSA_CRT_Q": "q",
        "RSA_CRT_DP": "dp",
        "RSA_CRT_DQ": "dq",
        "RSA_CRT_QINV": "qinv",
    }

    x_path = os.path.join(processed_dir, "X_features.npy")
    if not os.path.exists(x_path):
        logger.warning("RSA training skipped: missing %s", x_path)
        return False

    for comp, suffix in suffix_map.items():
        y_path = label_paths.get(comp)
        if not y_path or not os.path.exists(y_path):
            continue
        save_path = os.path.join(model_dir, f"best_model_rsa_{suffix}.pth")
        logger.info("Training RSA %s -> %s", comp, save_path)
        train_rsa_model(x_path, y_path, save_path, epochs=epochs, batch_size=batch_size)

    return True


def _resolve_rsa_model_root(model_root: str) -> str:
    if os.path.isdir(os.path.join(model_root, "rsa")):
        return model_root

    # Allow passing the ensemble directory directly.
    rsa_files = [
        os.path.join(model_root, "rsa_p_model0.pth"),
        os.path.join(model_root, "best_model_rsa_p.pth"),
    ]
    if any(os.path.exists(p) for p in rsa_files):
        return model_root

    fallback = os.path.join("models", "Ensemble_RSA")
    if os.path.isdir(fallback):
        logger.info("RSA models not found under %s. Using %s", model_root, fallback)
        return fallback

    return model_root


def _should_attempt_pin(meta_path: str) -> bool:
    try:
        df = pd.read_csv(meta_path, nrows=500)
    except Exception:
        return False
    for col in ("Verify_command", "Verify_response", "EncryptedPIN", "apdu", "IO", "C7"):
        if col not in df.columns:
            continue
        series = df[col].astype(str).str.replace(" ", "", regex=False).str.upper()
        if col in ("apdu", "IO", "C7"):
            if series.str.contains("0020").any():
                return True
        else:
            if series.str.len().gt(0).any():
                return True
    return False


def attack_rsa(
    processed_dir: str,
    model_root: str,
    meta_path: Optional[str] = None,
    run_pin: bool = True,
) -> Tuple[Optional[Dict[str, list]], Optional[str]]:
    x_path = os.path.join(processed_dir, "X_features.npy")
    if not os.path.exists(x_path):
        logger.warning("RSA attack skipped: missing %s", x_path)
        return None, None

    if meta_path is None:
        meta_path = os.path.join(processed_dir, "Y_meta.csv")

    model_dir = _resolve_rsa_model_root(model_root)
    rsa_predictions = attack_all_rsa_components(x_path, meta_path, model_dir)

    pin = None
    if run_pin and rsa_predictions and _should_attempt_pin(meta_path):
        pin = extract_pin_from_trace_rsa(meta_path, rsa_predictions)
    elif run_pin:
        logger.info("PIN extraction skipped: VERIFY/APDU not found in metadata.")

    return rsa_predictions, pin
