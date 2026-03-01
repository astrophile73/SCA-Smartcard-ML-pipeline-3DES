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
    
    # Try to find ensemble models first (support model_dir/rsa subfolder).
    if os.path.isdir(os.path.join(model_dir, "rsa")):
        model_dir = os.path.join(model_dir, "rsa")
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
            model = create_rsa_model(input_size=X.shape[1])
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
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
                if i == 0:
                    logger.info(f"Trace {i}: Derived consistent CRT components (P, Q, DP, DQ, QINV)")
    
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
