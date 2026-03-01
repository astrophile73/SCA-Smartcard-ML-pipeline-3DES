"""
RSA Training Script
Trains model to predict RSA key components
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from tqdm import tqdm

from src.model_rsa import create_rsa_model
from src.dataset_rsa import get_rsa_dataloaders

logger = logging.getLogger(__name__)

def train_rsa_model(X_path, Y_path, model_save_path, epochs=100, batch_size=32):
    """
    Train RSA model
    
    Args:
        X_path: Features path
        Y_path: Labels path (RSA component)
        model_save_path: Where to save best model
        epochs: Number of epochs
        batch_size: Batch size
    
    Returns:
        best_val_acc: Best validation accuracy achieved
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Load data
    train_loader, val_loader = get_rsa_dataloaders(X_path, Y_path, batch_size=batch_size)
    
    # Create model
    input_dim = next(iter(train_loader))[0].shape[1]
    model = create_rsa_model(input_size=input_dim)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 5
    perfect_score_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct_bytes = 0
        train_total_bytes = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # (batch, 128)
            
            optimizer.zero_grad()
            
            # Forward pass - get predictions for all 128 bytes
            outputs = model(batch_x)  # List of 128 tensors, each (batch, 256)
            
            # Calculate loss for each byte
            loss = 0
            for byte_idx in range(128):
                loss += criterion(outputs[byte_idx], batch_y[:, byte_idx])
            
            loss = loss / 128  # Average loss per byte
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy (how many bytes predicted correctly)
            for byte_idx in range(128):
                preds = torch.argmax(outputs[byte_idx], dim=1)
                train_correct_bytes += (preds == batch_y[:, byte_idx]).sum().item()
                train_total_bytes += batch_y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct_bytes / train_total_bytes
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_bytes = 0
        val_total_bytes = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                
                # Calculate loss
                loss = 0
                for byte_idx in range(128):
                    loss += criterion(outputs[byte_idx], batch_y[:, byte_idx])
                loss = loss / 128
                
                val_loss += loss.item()
                
                # Calculate accuracy
                for byte_idx in range(128):
                    preds = torch.argmax(outputs[byte_idx], dim=1)
                    val_correct_bytes += (preds == batch_y[:, byte_idx]).sum().item()
                    val_total_bytes += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct_bytes / val_total_bytes
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc >= 99.99:
            perfect_score_counter += 1
            if perfect_score_counter >= 3:
                logger.info(f"  [OK] Perfect Accuracy (100%) for 3 epochs. Stopping early!")
                break
        else:
            perfect_score_counter = 0

        # Save best model
        if val_acc >= best_val_acc: # Save even if 0.0 to ensure model exists
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"  [OK] Model saved to {model_save_path} (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping (standard)
        if patience_counter >= patience_limit:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            break
    
    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Train on RSA_CRT_P
    acc = train_rsa_model(
        "Processed/Mastercard/X_features.npy",
        "Processed/Mastercard/Y_labels_RSA_CRT_P.npy",
        "Optimization/best_model_rsa_p.pth",
        epochs=100,
        batch_size=32
    )
    
    print(f"Final accuracy: {acc:.2f}%")
