
import logging
from src.train_rsa import train_rsa_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Train RSA_CRT_DQ (Just to be sure, since it was last one running)
    print("Training RSA_CRT_DQ...")
    train_rsa_model(
        "Processed/Mastercard/X_features.npy",
        "Processed/Mastercard/Y_labels_RSA_CRT_DQ.npy", # Check specific label filename
        "Optimization/best_model_rsa_dq.pth",
        epochs=15, 
        batch_size=32
    )
    
    # Train RSA_CRT_QINV (Missing)
    print("Training RSA_CRT_QINV...")
    train_rsa_model(
        "Processed/Mastercard/X_features.npy",
        "Processed/Mastercard/Y_labels_RSA_CRT_QINV.npy", # Check specific label filename
        "Optimization/best_model_rsa_qinv.pth",
        epochs=15,
        batch_size=32
    )
