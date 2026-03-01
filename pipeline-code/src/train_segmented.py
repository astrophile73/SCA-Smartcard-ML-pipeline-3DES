
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.dataset_segmented import SegmentedSCADataset
from src.model import get_model
from src.utils import setup_logger

logger = setup_logger("train_segmented")

def train_round_sbox(x_path, y_path, model_out_path, epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(y_path):
        logger.warning(f"Label file {y_path} missing. Skipping.")
        return
        
    # Check if model exists
    if os.path.exists(model_out_path):
        logger.info(f"Model {model_out_path} already exists. Skipping.")
        return

    logger.info(f"Training on {y_path}...")
    
    # Load Dataset
    # We use FULL WINDOW (window=None) because feature_eng now covers 0-300k.
    # The CNN will learn to pick features.
    dataset = SegmentedSCADataset(x_path, y_path, window=None)
    
    # Split Train/Val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    # Input dim is 1500 (standard POIs)
    # Check actual dim
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]
    
    model = get_model(input_dim=input_dim, num_classes=16).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Added Regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    best_acc = 0.0
    patience = 15
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == by).sum().item()
            total += by.size(0)
            
        train_acc = correct / total
        
        # Val
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                _, preds = torch.max(out, 1)
                v_correct += (preds == by).sum().item()
                v_total += by.size(0)
        
        val_acc = v_correct / v_total
        scheduler.step(val_acc)
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_out_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
                
        if epoch % 5 == 0:
            logger.info(f"Ep {epoch}: Train {train_acc:.3f} | Val {val_acc:.3f}")
            
    logger.info(f"Finished. Best Val Acc: {best_acc:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="Processed/Mastercard")
    parser.add_argument("--opt_dir", default="Optimization")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rounds", nargs='+', default=['r1', 'r2', 'r3'], help="Rounds to train (r1 r2 r3)")
    args = parser.parse_args()
    
    start_sbox = 1 
    
    x_path = os.path.join(args.processed_dir, "X_features.npy")
    
    for r in args.rounds:
        for sb in range(1, 9):
            if r == 'r1' and sb < start_sbox: continue
            
            y_file = f"Y_labels_{r}_sbox{sb}.npy"
            y_path = os.path.join(args.processed_dir, y_file)
            
            model_file = f"best_model_{r}_sbox{sb}.pth"
            model_path = os.path.join(args.opt_dir, model_file)
            
            logger.info(f"=== Starting {r.upper()} SBox {sb} ===")
            train_round_sbox(x_path, y_path, model_path, epochs=args.epochs)

if __name__ == "__main__":
    main()
