"""
RSA Inference - Attack RSA keys from power traces
"""
import logging

import torch
import numpy as np
import pandas as pd
import warnings

# Suppress FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
from src.utils import setup_logger
logger = setup_logger("inference_rsa")
from src.model_rsa import create_rsa_model
#from Crypto.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from src.crypto import derive_rsa_crt

def perform_rsa_attack(features_path, meta_path, model_dir, component_name):
    """
    Attack RSA component using trained model (Optimized with Batching)
    
    Args:
        features_path: Path to X_features.npy
        meta_path: Path to Y_meta.csv
        model_dir: Directory containing RSA models
        component_name: Which component (RSA_CRT_P, RSA_CRT_Q, etc.)
    
    Returns:
        predictions: List of hex strings (one per trace)
    """
    try:
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        logger.error("torch.utils.data not available")
        return []

    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load features
    X = np.load(features_path).astype(np.float32)
    logger.info(f"Loaded {len(X)} traces for RSA attack on {component_name}")
    
    # Identify Models (Ensemble or Single)
    import glob
    import os
    comp_tag = component_name.split('_')[-1].lower()
    
    # Try to find ensemble models first (support model_dir/rsa or model_dir/Ensemble_RSA subfolder).
    if os.path.isdir(os.path.join(model_dir, "rsa")):
        model_dir = os.path.join(model_dir, "rsa")
    elif os.path.isdir(os.path.join(model_dir, "Ensemble_RSA")):
        model_dir = os.path.join(model_dir, "Ensemble_RSA")
    
    ensemble_pattern = os.path.join(model_dir, f"rsa_{comp_tag}_model*.pth")
    model_paths = sorted(glob.glob(ensemble_pattern))
    
    # Fallback to single best model
    if not model_paths:
        single_path = os.path.join(model_dir, f"best_model_rsa_{comp_tag}.pth")
        if os.path.exists(single_path):
            model_paths = [single_path]
        else:
            logger.error(f"No models found for {component_name} in {model_dir}")
            return [''] * len(X)

    logger.info(f"Using {len(model_paths)} models for {component_name} inference.")

    # Load All Models
    models = []
    for p in model_paths:
        try:
            # Load checkpoint first to infer expected input dimension
            checkpoint = torch.load(p, map_location=device, weights_only=False)
            
            # Infer input_size from the first linear layer's weight shape
            # shared_features.0.weight has shape (2048, input_size)
            inferred_input_size = None
            if "shared_features.0.weight" in checkpoint:
                inferred_input_size = checkpoint["shared_features.0.weight"].shape[1]
            
            # Use inferred size if available, otherwise fall back to current feature dimension
            input_size = inferred_input_size if inferred_input_size is not None else X.shape[1]
            
            logger.debug(f"Creating model with input_size={input_size} (inferred={inferred_input_size}, current_features={X.shape[1]})")
            
            model = create_rsa_model(input_size=input_size)
            model.load_state_dict(checkpoint, strict=True)
            model = model.to(device)
            model.eval()
            models.append(model)
        except Exception as e:
            logger.error(f"Failed to load model {p}: {e}")
    
    if not models:
        return [''] * len(X)
    
    # Setup Data Loader
    batch_size = 512
    dataset = TensorDataset(torch.from_numpy(X))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Perform inference
    predictions = []
    
    logger.info(f"Starting ensemble inference with batch size {batch_size}...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x_batch = batch[0].to(device)
            
            # Aggregate probabilities from all models
            summed_logits = None
            
            for model in models:
                # Model returns list of 128 tensors, each (Batch, 256)
                outputs = model(x_batch) 
                # Stack to (Batch, 128, 256)
                stacked_outputs = torch.stack(outputs, dim=1)
                
                if summed_logits is None:
                    summed_logits = stacked_outputs
                else:
                    summed_logits += stacked_outputs
            
            # Average (optional, argmax is same for sum or avg)
            avg_logits = summed_logits / len(models)
            
            # Get max probability indices
            predicted_indices = torch.argmax(avg_logits, dim=2).cpu().numpy()
            
            # Convert to hex strings
            for row in predicted_indices:
                hex_str = ''.join(f'{b:02X}' for b in row)
                predictions.append(hex_str)
            
            if i % 10 == 0:
                logger.debug(f"Processed batch {i}/{len(dataloader)}")

    logger.info(f"RSA ensemble attack complete for {component_name}")
    if predictions:
        logger.info(f"Sample prediction (first 60 chars): {predictions[0][:60]}")
    
    return predictions

def attack_all_rsa_components(features_path, meta_path, model_dir):
    """
    Attack all 5 RSA components
    
    Returns:
        dict: {component_name: [predictions]}
    """
    components = ['RSA_CRT_P', 'RSA_CRT_Q', 'RSA_CRT_DP', 'RSA_CRT_DQ', 'RSA_CRT_QINV']
    
    results = {}
    
    for comp in components:
        logger.info(f"Attacking {comp}...")
        predictions = perform_rsa_attack(features_path, meta_path, model_dir, comp)
        results[comp] = predictions
    
    # RSATOOL LOGIC: Validate consistency and fix INVq if P and Q are strong
    logger.info("Verifying RSA consistency via rsatool logic...")
    n_preds = len(results['RSA_CRT_P'])
    
    # Helper to strip trailing zero bytes from 128-byte zero-padded hex strings
    def strip_rsa_padding(hex_str):
        """Strip trailing '00' pairs from RSA predictions that are zero-padded to 256 chars (128 bytes)."""
        if not hex_str or len(hex_str) < 2:
            return hex_str
        while hex_str.endswith('00') and len(hex_str) > 2:
            hex_str = hex_str[:-2]
        return hex_str
    
    derive_success = 0
    derive_fail = 0
    
    for i in range(n_preds):
        p_hex = results['RSA_CRT_P'][i]
        q_hex = results['RSA_CRT_Q'][i]
        
        if p_hex and q_hex:
            derived = derive_rsa_crt(p_hex, q_hex)
            if derived:
                # Always use derived values for consistency
                results['RSA_CRT_P'][i] = derived['P']
                results['RSA_CRT_Q'][i] = derived['Q']
                results['RSA_CRT_QINV'][i] = derived['QINV']
                results['RSA_CRT_DP'][i] = derived['DP']
                results['RSA_CRT_DQ'][i] = derived['DQ']
                derive_success += 1
                if i == 0:
                    logger.info(f"Trace {i}: Derived consistent CRT components (P, Q, DP, DQ, QINV)")
            else:
                # Derive failed: strip padding from original predictions as fallback
                logger.debug(f"Trace {i}: derive_rsa_crt failed for valid P/Q; using padded originals with trailing zeros stripped")
                results['RSA_CRT_P'][i] = strip_rsa_padding(p_hex)
                results['RSA_CRT_Q'][i] = strip_rsa_padding(q_hex)
                results['RSA_CRT_DP'][i] = strip_rsa_padding(results['RSA_CRT_DP'][i])
                results['RSA_CRT_DQ'][i] = strip_rsa_padding(results['RSA_CRT_DQ'][i])
                results['RSA_CRT_QINV'][i] = strip_rsa_padding(results['RSA_CRT_QINV'][i])
                derive_fail += 1
        else:
            # At least one of P or Q is empty: strip padding from all components as fallback
            if p_hex or q_hex:
                logger.debug(f"Trace {i}: One of P/Q is missing; stripping trailing zeros from all components")
            results['RSA_CRT_P'][i] = strip_rsa_padding(p_hex)
            results['RSA_CRT_Q'][i] = strip_rsa_padding(q_hex)
            results['RSA_CRT_DP'][i] = strip_rsa_padding(results['RSA_CRT_DP'][i])
            results['RSA_CRT_DQ'][i] = strip_rsa_padding(results['RSA_CRT_DQ'][i])
            results['RSA_CRT_QINV'][i] = strip_rsa_padding(results['RSA_CRT_QINV'][i])
            if not p_hex and not q_hex:
                derive_fail += 1
    
    logger.info(f"RSA consistency check: {derive_success} traces with derived values, {derive_fail} traces with padding stripped as fallback")
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    results = attack_all_rsa_components(
        "Processed/X_features.npy",
        "Processed/Y_meta.csv",
        "Optimization"
    )
    
    print("\nRSA Attack Results:")
    for comp, preds in results.items():
        print(f"{comp}: {len(preds)} predictions")
        print(f"  Sample (60 chars): {preds[0][:60]}")
