"""
Week 1 Improved Training Script - Better handling of class imbalance

Strategies:
1. Aggressive class weighting (inverse frequency)
2. Focal loss for difficult samples
3. Data balancing (oversampling minority classes)
4. Label smoothing
5. Lower initial learning rate
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from model_week1 import ASCADModel
from torch.optim.lr_scheduler import CosineAnnealingLR


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=1, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class Week1ImprovedTrainer:
    """Improved trainer with better handling of class imbalance"""
    
    def __init__(self, sbox_idx=0, max_epochs=100, batch_size=32, no_cuda=True):
        self.sbox_idx = sbox_idx
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')
        
        # Paths
        self.data_dir = Path(__file__).parent.parent / "Processed" / "3des"
        self.model_dir = Path(__file__).parent.parent / "models" / "3des"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_output_dir = self.model_dir / f"week1_improved_{self.timestamp}"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Model output: {self.model_output_dir}")
        
    def load_data(self):
        """Load training data"""
        print(f"\n{'='*70}")
        print("Loading Data")
        print(f"{'='*70}\n")
        
        # Load features
        X_path = self.data_dir / "X_features.npy"
        if not X_path.exists():
            raise FileNotFoundError(f"Features not found: {X_path}")
        
        X = np.load(X_path).astype(np.float32)
        print(f"Features shape: {X.shape}")
        
        # Load labels
        label_path = self.data_dir / f"Y_labels_sbox_input_kenc_s1_sbox{self.sbox_idx+1}.npy"
        if not label_path.exists():
            raise FileNotFoundError(f"Labels not found: {label_path}")
        
        y = np.load(label_path).astype(np.int64)
        print(f"Labels shape: {y.shape}")
        print(f"Unique classes: {len(np.unique(y))} / 64")
        print(f"Label range: [{y.min()}, {y.max()}]")
        
        # Analyze class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / len(y)
            print(f"  Class {label:2d}: {count:5d} samples ({pct:5.1f}%)")
        
        return X, y, unique, counts
    
    def create_balanced_sampler(self, y):
        """Create weighted sampler for balanced sampling"""
        unique, counts = np.unique(y, return_counts=True)
        
        # Inverse frequency weights
        class_weights = {}
        total_samples = len(y)
        for cls, count in zip(unique, counts):
            weight = total_samples / (len(unique) * count)
            class_weights[cls] = weight
        
        sample_weights = np.array([class_weights[label] for label in y])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y),
            replacement=True
        )
        return sampler
    
    def train(self):
        """Main training loop"""
        # Load data
        X, y, unique, counts = self.load_data()
        
        # Split data (80/20)
        n_train = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )
        
        # Create samplers for balanced training
        train_sampler = self.create_balanced_sampler(y_train)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Model
        num_classes = max(64, y.max() + 1)  # At least 64, but handle if higher
        model = ASCADModel(input_dim=X.shape[1], num_classes=num_classes).to(self.device)
        
        # Loss function: Focal Loss + Label Smoothing
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Optimizer: lower learning rate for stability
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        
        # Training loop
        print(f"\n{'='*70}")
        print(f"Training (Focal Loss + Weighted Sampling)")
        print(f"{'='*70}\n")
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(self.max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == y_batch).sum().item()
                    val_total += y_batch.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{self.max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                best_model_path = self.model_output_dir / "model_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(" (BEST)")
            else:
                patience_counter += 1
                print()
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
                    break
            
            scheduler.step()
        
        print(f"\n{'='*70}")
        print(f"Training Complete")
        print(f"{'='*70}")
        print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")
        print(f"Model saved to: {self.model_output_dir / 'model_best.pth'}")
        
        return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Week 1 Improved Training')
    parser.add_argument('--sbox', type=int, default=0, help='S-Box index (0-7)')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    trainer = Week1ImprovedTrainer(
        sbox_idx=args.sbox,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        no_cuda=args.no_cuda
    )
    
    model, acc = trainer.train()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
