import torch
from torch.utils.data import Dataset
import numpy as np
import os
from src.utils import setup_logger

logger = setup_logger("dataset")

class SCADataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"{x_path} not found")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"{y_path} not found")
            
        self.X = np.load(x_path).astype(np.float32)
        self.Y = np.load(y_path).astype(np.longlong) # Labels for CrossEntropyLoss
        
        # Check shapes
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"Mismatch in samples: X={self.X.shape[0]}, Y={self.Y.shape[0]}")
            
        self.transform = transform
        self.enable_noise = False
        self.noise_std = 0.1
        
        # Generalization: Data augmentation (ENABLED for better generalization)
        self.enable_augmentation = True
        self.augmentation_strength = 0.1  # Amplitude scaling range
        
        logger.info(f"Loaded dataset: X={self.X.shape}, Y={self.Y.shape}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.enable_noise:
            # Gaussian Noise
            noise = np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
            x = x + noise
        
        if self.enable_augmentation:
            # Amplitude scaling for generalization
            scale = 1.0 + np.random.uniform(-self.augmentation_strength, self.augmentation_strength)
            x = x * scale
        
        if self.transform:
            x = self.transform(x)
            
        return torch.from_numpy(x), torch.tensor(y)

def get_dataloaders(x_path, y_path, batch_size=64, val_split=0.2, seed=42):
    from torch.utils.data import DataLoader, random_split, Subset
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    # We use sklearn for splitting indices to acturally stratify if needed, 
    # but random split is fine for now.
    
    full_dataset = SCADataset(x_path, y_path)
    
    # Stratified split is better for SCA
    targets = full_dataset.Y
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        random_state=seed,
        stratify=targets # Important because of class imbalance
    )
    
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
