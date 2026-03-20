"""
Feature-Label Validation & Regeneration Module

Ensures preprocessing is always consistent with the current label configuration.
Prevents feature-label mismatches by:
1. Storing metadata about which labels were used for POI selection
2. Validating that cached features match current label type
3. Forcing regeneration if label type changed
"""

import os
import json
import numpy as np
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger("preprocess_validator")


def should_regenerate_features(output_dir, current_label_type="sbox_output", verbose=True):
    """
    Check if features should be regenerated.
    
    Returns True if:
    - Features don't exist
    - Label type changed
    - Metadata file corrupted
    
    Args:
        output_dir: Directory containing preprocessed features
        current_label_type: Expected label type (sbox_output or sbox_input)
        verbose: Print detailed messages
    
    Returns:
        bool: True if regeneration needed, False if can reuse
    """
    config_file = os.path.join(output_dir, "preprocessing_config.json")
    meta_file = os.path.join(output_dir, "Y_meta.csv")
    feature_file = os.path.join(output_dir, "X_features.npy")
    
    # Check if basic files exist
    if not os.path.exists(meta_file) or not os.path.exists(feature_file):
        if verbose:
            logger.info(f"[VALIDATION] Features don't exist, will regenerate")
        return True
    
    # Check if config exists and label type matches
    if not os.path.exists(config_file):
        if verbose:
            logger.warning(f"[VALIDATION] No preprocessing config found. Assuming old extraction. Will regenerate to ensure feature-label match.")
        return True
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        if verbose:
            logger.error(f"[VALIDATION] Could not read config file: {e}. Will regenerate.")
        return True
    
    # Check if label type changed
    last_label_type = config.get("label_type", "sbox_output")
    
    if last_label_type != current_label_type:
        if verbose:
            logger.warning(
                f"[VALIDATION] ⚠️  CRITICAL: Label type changed!\n"
                f"    Last extraction: {last_label_type}\n"
                f"    Current request: {current_label_type}\n"
                f"    → Features must be REGENERATED to avoid feature-label mismatch!"
            )
        return True
    
    if verbose:
        logger.info(f"[VALIDATION] [OK] Features valid for label_type='{current_label_type}'")
    
    return False


def save_preprocess_config(output_dir, label_type):
    """
    Save preprocessing metadata for future validation.
    
    This ensures we can detect when label type changed and force regeneration.
    """
    config_file = os.path.join(output_dir, "preprocessing_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "label_type": label_type,
        "version": "1.0",  # For future backward compatibility
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"[CONFIG] Saved: label_type={label_type}")
    except Exception as e:
        logger.error(f"[CONFIG] Could not save preprocessing config: {e}")
