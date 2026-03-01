"""
RSA Label Generator for SCA Pipeline
Generates training labels for RSA key recovery
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def generate_rsa_labels(meta_df, ref_keys_df=None, output_dir="Processed"):
    """
    Generate RSA key component labels for training (pure mode).

    This function intentionally does NOT require external spreadsheets.
    If RSA ground truth columns are present inside `meta_df`, they are used.
    Otherwise labels are saved as empty strings and RSA training should be skipped.
    
    Args:
        meta_df: Metadata DataFrame with Track2
        ref_keys_df: Deprecated (ignored). Kept for backward compatibility.
        output_dir: Output directory for labels
    
    Returns:
        Paths to generated label files
    """
    logger.info("Generating RSA labels...")
    
    # Mapping from Excel Column -> Output Suffix
    # main.py expects: Y_labels_rsa_p.npy, etc.
    component_map = {
        'RSA_CRT_P': 'p',
        'RSA_CRT_Q': 'q', 
        'RSA_CRT_DP': 'dp',
        'RSA_CRT_DQ': 'dq',
        'RSA_CRT_QINV': 'qinv'
    }
    
    # Initialize label storage
    labels = {comp: [] for comp in component_map.keys()}
    
    for idx, row in meta_df.iterrows():
        for comp in component_map.keys():
            # Pure mode: use labels only if they exist in meta_df itself.
            hex_val = str(row.get(comp, "")).strip()
            if hex_val and hex_val.lower() != "nan":
                labels[comp].append(hex_val)
            else:
                labels[comp].append("")
    
    # Save labels
    label_paths = {}
    for comp, suffix in component_map.items():
        # Convert to numpy array of strings
        label_array = np.array(labels[comp], dtype=object)
        # Output format: Y_labels_rsa_p.npy
        output_path = os.path.join(output_dir, f'Y_labels_rsa_{suffix}.npy')
        np.save(output_path, label_array)
        label_paths[comp] = output_path
        logger.info(f"Saved {suffix} labels: {output_path}")
    
    return label_paths

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    meta_df = pd.read_csv("Processed/Mastercard/Y_meta.csv")
    paths = generate_rsa_labels(meta_df, None, "Processed")
    print("Generated label files:", paths)
