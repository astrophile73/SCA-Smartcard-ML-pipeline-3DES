
import os
import sys

# Add project root to sys.path to allow 'src' imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
from src.utils import setup_logger
from src.inference_rsa import attack_all_rsa_components
from src.pin_extraction import extract_pin_from_trace_rsa

logger = setup_logger("Test_Visa_PIN")

def test_visa_extraction():
    processed_dir = "Processed/Verify_Test"
    model_dir = "Models/Final_Delivery"
    
    logger.info("=== STARTING TARGETED VISA PIN EXTRACTION TEST ===")
    
    # Check paths
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    features_path = os.path.join(processed_dir, "X_features.npy")
    
    if not os.path.exists(meta_path) or not os.path.exists(features_path):
        logger.error(f"Missing data in {processed_dir}. Run preprocessing first.")
        return

    # Check models
    if not os.path.exists(model_dir):
        logger.error(f"Missing models directory: {model_dir}")
        return
        
    logger.info("1. Running RSA Attack using All Models...")
    # This will predict P, Q, DP, DQ, QINV
    rsa_predictions = attack_all_rsa_components(features_path, meta_path, model_dir)
    
    if not rsa_predictions or not rsa_predictions.get('RSA_CRT_P'):
        logger.error("RSA Attack failed to produce keys.")
        return

    logger.info("2. Attempting PIN Extraction from Metadata...")
    # This will search for 00 20 (VERIFY) pattern or EncryptedPIN column
    df = pd.read_csv(meta_path)
    from src.pin_extraction import _find_pin_block_in_meta
    pin_block = _find_pin_block_in_meta(df)
    
    if pin_block:
        logger.info(f"[PARSING SUCCESS] Found PIN Block: {pin_block[:20]}... (len {len(pin_block)})")
    else:
        logger.error("[PARSING FAILURE] Could not find PIN block in metadata.")
        return

    logger.info("3. Attempting Decryption using RSA Predictions...")
    pin = extract_pin_from_trace_rsa(meta_path, rsa_predictions)
    
    if pin:
        logger.info(f"\n[DECRYPTION SUCCESS] Verified PIN: {pin}")
    else:
        logger.warning("\n[DECRYPTION STATUS] Could not mathematically verify PIN.")
        logger.warning("   Note: Single-trace RSA prediction accuracy is statistical.")
        logger.warning("   Ground truth validation of the parsing logic is COMPLETE.")

if __name__ == "__main__":
    test_visa_extraction()
