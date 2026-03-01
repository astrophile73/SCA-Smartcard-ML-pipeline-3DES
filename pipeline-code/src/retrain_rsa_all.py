
import os
import logging
from src.train_rsa import train_rsa_model
from src.utils import setup_logger

logger = setup_logger("retrain_rsa")

def retrain_all():
    components = ['RSA_CRT_P', 'RSA_CRT_Q', 'RSA_CRT_DP', 'RSA_CRT_DQ', 'RSA_CRT_QINV']
    
    # Ensure directories exist
    os.makedirs("Optimization", exist_ok=True)
    
    for comp in components:
        logger.info(f"=== Retraining Model for {comp} ===")
        
        # Construct paths
        # Y labels file
        y_path = f"Processed/Mastercard/Y_labels_{comp}.npy"
        
        # Check if exists
        if not os.path.exists(y_path):
            logger.error(f"Label file not found: {y_path}")
            continue
            
        # Common Feature file
        x_path = "Processed/Mastercard/X_features.npy"
        
        # Output Model Path
        # Naming convention: best_model_rsa_p.pth (lowercase, suffix only)
        # comp: RSA_CRT_P -> p
        suffix = comp.split('_')[-1].lower() # P, Q, DP...
        # handle QINV -> qinv
        
        model_name = f"best_model_rsa_{suffix}.pth"
        model_path = os.path.join("Optimization", model_name)
        
        # Train
        # We use a small number of epochs because previous logs showed it converged in < 5 epochs
        # But user wants "100%", so let's give it 20 epochs just in case.
        
        try:
            acc = train_rsa_model(
                x_path,
                y_path,
                model_path,
                epochs=20, # Reduced from 100 since it converges instantly
                batch_size=32
            )
            logger.info(f"Model {comp} saved to {model_path} with Acc: {acc:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to train {comp}: {e}")

if __name__ == "__main__":
    retrain_all()
