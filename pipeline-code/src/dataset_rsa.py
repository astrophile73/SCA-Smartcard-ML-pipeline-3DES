"""
RSA Dataset for PyTorch training
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class RSADataset(Dataset):
    """Dataset for RSA key recovery"""
    
    def __init__(self, X_path, Y_path):
        """
        Args:
            X_path: Path to features (.npy)
            Y_path: Path to RSA component labels (.npy)
        """
        # Convert hex strings to byte arrays and filter out dummies
        self.X_full = np.load(X_path).astype(np.float32)
        self.Y_hex_full = np.load(Y_path, allow_pickle=True)
        
        valid_indices = []
        Y_bytes = []
        
        for idx, hex_str in enumerate(self.Y_hex_full):
            # A dummy/invalid label is empty, 'nan', or '00...0' (usually 256 chars for RSA 1024)
            h_str = str(hex_str).strip()
            if not h_str or h_str.lower() == 'nan' or all(c == '0' for c in h_str):
                continue
            
            # Pad to 256 chars (128 bytes)
            hex_clean = h_str.ljust(256, '0')[:256]
            try:
                byte_array = [int(hex_clean[i:i+2], 16) for i in range(0, 256, 2)]
                Y_bytes.append(byte_array)
                valid_indices.append(idx)
            except:
                continue
        
        if not valid_indices:
             # Fallback if everything is filtered (should not happen for training sets)
             self.X = self.X_full
             self.Y = np.zeros((len(self.X_full), 128), dtype=np.int64)
             print(f"WARNING: No valid RSA labels found in {Y_path}")
        else:
             self.X = self.X_full[valid_indices]
             self.Y = np.array(Y_bytes, dtype=np.int64)
             print(f"Loaded filtered RSA dataset: X={self.X.shape}, Y={self.Y.shape} (Removed {len(self.X_full) - len(valid_indices)} dummy samples)")
        
        print(f"Loaded RSA dataset: X={self.X.shape}, Y={self.Y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])  # (128,) - one label per byte
        return x, y

def get_rsa_dataloaders(X_path, Y_path, batch_size=32, val_split=0.2):
    """
    Create train/val dataloaders for RSA
    
    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    dataset = RSADataset(X_path, Y_path)
    
    # Split indices
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=42,
        shuffle=True
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test
    train_loader, val_loader = get_rsa_dataloaders(
        "Processed/X_features.npy",
        "Processed/Y_labels_RSA_CRT_P.npy",
        batch_size=32
    )
    
    # Test batch
    x, y = next(iter(train_loader))
    print(f"Batch: X={x.shape}, Y={y.shape}")
    print(f"Y sample (first 10 bytes): {y[0][:10].tolist()}")
