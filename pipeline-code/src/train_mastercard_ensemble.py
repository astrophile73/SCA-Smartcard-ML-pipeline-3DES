import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Add project root to path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import SCADataset
from src.model_zaid import get_model
from src.utils import setup_logger
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = setup_logger("train_mastercard_ensemble")

def train_single_model(X, Y, model_idx, sbox_idx, save_dir, epochs=50, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Stratified Split
    try:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42 + model_idx, stratify=Y)
    except ValueError:
        # Fallback if class imbalance prevents stratification
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42 + model_idx)

    # Tensor conversion
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = get_model(input_dim=X.shape[1], num_classes=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    save_path = os.path.join(save_dir, f"sbox{sbox_idx}_model{model_idx}.pth")

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / (total + 1e-10)
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    logger.info(f"S-Box {sbox_idx} Model {model_idx} Finished. Best Acc: {best_acc:.2f}%")
    return best_acc

def train_ensemble(input_dir, output_dir, models_per_sbox=5, epochs=50):
    os.makedirs(output_dir, exist_ok=True)
    
    x_path = os.path.join(input_dir, "X_features.npy")
    if not os.path.exists(x_path):
        logger.error(f"Features not found at {x_path}")
        return

    logger.info(f"Loading features from {x_path}...")
    X = np.load(x_path).astype(np.float32)
    
    # Z-Score Normalization
    logger.info("Normalizing features...")
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1 
    X = (X - mean) / std
    
    # Save statistics for inference normalization
    np.save(os.path.join(output_dir, "mean.npy"), mean)
    np.save(os.path.join(output_dir, "std.npy"), std)
    
    for sbox in range(8):
        y_path = os.path.join(input_dir, f"Y_labels_sbox{sbox+1}.npy")
        if not os.path.exists(y_path):
            logger.warning(f"Labels not found for SBox {sbox+1}")
            continue

        Y = np.load(y_path).astype(np.longlong)
        
        # Filter -1 (Invalid labels)
        valid_idx = Y != -1
        X_curr = X[valid_idx]
        Y_curr = Y[valid_idx]
        
        if len(Y_curr) == 0:
            logger.warning(f"No valid labels for SBox {sbox+1}")
            continue

        logger.info(f"Training Ensemble for S-Box {sbox+1} ({models_per_sbox} models)...")
        
        for i in range(models_per_sbox):
            train_single_model(X_curr, Y_curr, i, sbox+1, output_dir, epochs=epochs)

if __name__ == "__main__":
    # Hardcoded paths for the Master Key strategy
    input_dir = "Processed/Mastercard_ATC"
    output_dir = "Models/Ensemble_MasterKey_Visa"
    train_ensemble(input_dir, output_dir, models_per_sbox=3, epochs=30) # 3 models/sbox, 30 epochs for speed/efficacy balance
