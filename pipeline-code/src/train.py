import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import setup_logger

logger = setup_logger("train")

def train(x_path, y_path, epochs=10, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data
    train_loader, val_loader = get_dataloaders(x_path, y_path, batch_size=batch_size)
    
    # Model
    model = get_model(input_dim=200, num_classes=16).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    best_acc = 0.0
    
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        # Train
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
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        
        # Validation
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
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "Optimization/best_model.pth")
            logger.info("Saved new best model.")
            
    return best_acc

if __name__ == "__main__":
    train(
        "Processed/X_features.npy",
        "Processed/Y_labels_sbox1.npy",
        epochs=5 # Quick dry run for Day 6
    )
