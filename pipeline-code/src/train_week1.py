"""
Week 1 Training Script - Fixed 3DES ML Pipeline

This script retrains the 3DES attack model with the corrected approach:
- Uses S-Box INPUT labels (6-bit, 0-63) instead of S-Box OUTPUT (4-bit, 0-15)
- Model output: 64 classes (for 6-bit S-Box inputs)
- Implements class weighting for imbalanced data
- Better accuracy and key recovery capability

To run:
python train_week1.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import argparse
from datetime import datetime

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from model_week1 import ASCADModel, compute_class_weights
from gen_labels_sbox_input import compute_sbox_input_labels


class Week1Trainer:
    """Trainer for fixed 3DES model with S-Box input labels"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
        )
        print(f"Device: {self.device}")
        
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = Path(args.model_dir) / f"week1_{self.timestamp}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load training and test data with S-Box input labels"""
        print(f"\n{'='*60}")
        print("Loading data...")
        print(f"{'='*60}")
        
        data_dir = Path(self.args.data_dir)
        
        # Load features
        X_train_path = data_dir / "X_features.npy"
        if not X_train_path.exists():
            print(f"❌ Training features not found: {X_train_path}")
            return False
        
        X_train = np.load(X_train_path).astype(np.float32)
        print(f"[OK] Training features: {X_train.shape}")
        
        # Load labels - try new S-Box input labels first
        sbox_idx = self.args.sbox_idx
        label_patterns = [
            data_dir / f"Y_labels_sbox_input_kenc_s1_sbox{sbox_idx+1}.npy",
            data_dir / f"Y_labels_sbox{sbox_idx+1}.npy",  # Fallback to old labels
        ]
        
        y_train = None
        label_file = None
        for pattern in label_patterns:
            if pattern.exists():
                y_train = np.load(pattern).astype(np.int64)
                label_file = pattern
                break
        
        if y_train is None:
            print(f"❌ No labels found for S-Box {sbox_idx}")
            return False
        
        print(f"[OK] Labels: {y_train.shape} from {label_file.name}")
        print(f"  Unique values: {len(np.unique(y_train[y_train >= 0]))}")
        print(f"  Range: [{y_train[y_train >= 0].min()}, {y_train[y_train >= 0].max()}]")
        
        # Remove invalid labels (-1)
        valid_mask = y_train >= 0
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        print(f"[OK] After removing invalid labels: X={X_train.shape}, y={y_train.shape}")
        
        # Split into train/val
        n_samples = len(X_train)
        train_size = int(0.8 * n_samples)
        
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        
        X_train_split = X_train[train_idx]
        y_train_split = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        print(f"[OK] Train split: {X_train_split.shape}, {y_train_split.shape}")
        print(f"[OK] Val split: {X_val.shape}, {y_val.shape}")
        
        return X_train_split, y_train_split, X_val, y_val
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val):
        """Create PyTorch DataLoaders"""
        print(f"\nCreating DataLoaders...")
        
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"[OK] Train batches: {len(train_loader)}")
        print(f"[OK] Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("WEEK 1 TRAINING - Fixed 3DES Model")
        print(f"{'='*60}")
        
        # Load data
        data = self.load_data()
        if not data:
            return False
        X_train, y_train, X_val, y_val = data
        
        # Create model
        print(f"\n{'='*60}")
        print("Creating model...")
        print(f"{'='*60}")
        
        model = ASCADModel(
            input_dim=X_train.shape[1],
            num_classes=64  # 6-bit S-Box inputs
        ).to(self.device)
        
        print(f"[OK] Model created")
        print(f"  Input dim: {X_train.shape[1]}")
        print(f"  Output classes: 64 (S-Box inputs)")
        
        # Compute class weights
        class_weights = compute_class_weights(y_train, num_classes=64).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs
        )
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(X_train, y_train, X_val, y_val)
        
        # Training loop
        print(f"\n{'='*60}")
        print(f"Training for {self.args.epochs} epochs...")
        print(f"{'='*60}\n")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(1, self.args.epochs + 1):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            scheduler.step()
            
            # Print progress
            if epoch % max(1, self.args.epochs // 10) == 0 or epoch <= 5:
                print(f"Epoch {epoch:3d}/{self.args.epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                model_path = self.model_dir / f"model_best.pth"
                torch.save(model.state_dict(), model_path)
                print(f"  [OK] Best model saved (acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    print(f"\n[OK] Early stopping at epoch {epoch}")
                    break
        
        # Save final model
        final_path = self.model_dir / f"model_final.pth"
        torch.save(model.state_dict(), final_path)
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"{'='*60}")
        print(f"[OK] Best validation accuracy: {best_val_acc:.4f}")
        print(f"[OK] Models saved in: {self.model_dir}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Week 1 Training - Fixed 3DES ML Pipeline")
    
    parser.add_argument("--data-dir", default="3des-pipeline/Processed/3des",
                       help="Directory with training features and labels")
    parser.add_argument("--model-dir", default="pipeline-code/models/3des",
                       help="Directory to save models")
    parser.add_argument("--sbox-idx", type=int, default=0,
                       help="S-Box index (0-7)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--no-cuda", action="store_true",
                       help="Disable CUDA")
    
    args = parser.parse_args()
    
    trainer = Week1Trainer(args)
    success = trainer.train()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
