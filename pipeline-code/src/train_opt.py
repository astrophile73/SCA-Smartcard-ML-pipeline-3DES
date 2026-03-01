import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from src.dataset import SCADataset
from src.model_zaid import get_model
from src.utils import setup_logger

logger = setup_logger("train_opt")

def get_weighted_loader(dataset, subset_indices, batch_size=64):
    """
    Create a DataLoader with WeightedRandomSampler to handle class imbalance.
    """
    # Get labels for the subset
    targets = dataset.Y[subset_indices]
    
    # Calculate class counts
    class_counts = np.bincount(targets, minlength=16)
    
    # Avoid div by zero
    class_counts = class_counts.astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    
    # Weights are inverse of frequency
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample
    sample_weights = class_weights[targets]
    
    # Sampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        Subset(dataset, subset_indices),
        batch_size=batch_size,
        sampler=sampler, # Replaces shuffle=True
        drop_last=True
    )

def train_optimized(X_path, Y_path, epochs=50, batch_size=64, learning_rate=0.0005, save_path="Optimization/best_model_opt.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    full_dataset = SCADataset(X_path, Y_path)
    
    # Filter out samples with label -1 (Dummy Keys)
    valid_mask = full_dataset.Y != -1
    if not np.any(valid_mask):
        logger.error(f"No valid samples found in {Y_path} (all are -1).")
        return 0.0
        
    valid_indices = np.where(valid_mask)[0]
    full_dataset.X = full_dataset.X[valid_indices]
    full_dataset.Y = full_dataset.Y[valid_indices]
    
    logger.info(f"Filtered dataset: {len(full_dataset)} valid samples remaining.")
    
    indices = np.arange(len(full_dataset))
    targets = full_dataset.Y
    
    # Check if stratification is possible
    unique, counts = np.unique(targets, return_counts=True)
    can_stratify = all(counts >= 2)
    
    # Stratified Split (if possible)
    if can_stratify:
        logger.info("Using stratified train/test split")
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=targets
        )
    else:
        logger.warning(f"Cannot stratify: some classes have <2 samples. Using random split.")
        logger.warning(f"Class distribution: {dict(zip(unique, counts))}")
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42
        )
    
    # Loaders with Weighted Sampling
    train_loader = get_weighted_loader(full_dataset, train_idx, batch_size)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    input_dim = full_dataset.X.shape[1]
    model = get_model(input_dim=input_dim, num_classes=16).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Added Regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    no_improve_epochs = 0
    no_improve_epochs = 0
    patience_limit = 200 # Disabled early stopping practically
    
    for epoch in range(epochs):
        # Train with augmentation for generalization
        full_dataset.enable_augmentation = True
        # full_dataset.enable_noise = True  # DISABLED for testing
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if total > 0:
            train_loss = train_loss / total
            train_acc = 100 * correct / total
        else:
            train_loss = 0
            train_acc = 0
        
        # Validation (no augmentation)
        full_dataset.enable_augmentation = False
        # full_dataset.enable_noise = False  # DISABLED for testing
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / total
        val_acc = 100 * correct / total
        
        # Logging
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_acc)
        
        if val_acc >= best_acc: # Use >= to save even if 0.0 initially
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            no_improve_epochs = 0
            # logger.debug(f"Saved improved model to {save_path}")
        else:
            no_improve_epochs += 1
            
        if val_acc >= 99.0:
            logger.info("Target Accuracy (>=99%) Reached!")
            break
            
        if no_improve_epochs >= patience_limit:
            logger.info("Early Stopping triggered.")
            break
            
    logger.info(f"Finished. Best Val Acc: {best_acc:.2f}%")
    return best_acc

if __name__ == "__main__":
    train_optimized(
        "Processed/X_features.npy",
        "Processed/Y_labels_sbox1.npy",
        epochs=100
    )
