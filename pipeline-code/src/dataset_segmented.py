import torch
from torch.utils.data import Dataset
import numpy as np
import os
from src.utils import setup_logger

logger = setup_logger("dataset_segmented")

class SegmentedSCADataset(Dataset):
    def __init__(self, x_path, y_path, window=None):
        """
        Args:
            x_path: Path to .npy file containing traces [N, Time]
            y_path: Path to labels [N]
            window: Tuple (start_index, end_index) or None
        """
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"{x_path} not found")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"{y_path} not found")
            
        logger.info(f"Loading data from {x_path}")
        # Load X
        # Use mmap_mode='r' to avoid loading everything if we are going to slice immediately
        # But slicing creates a copy anyway.
        # If dataset is small (20k traces), loading full is fine.
        X_full = np.load(x_path, mmap_mode='r')
        
        if window:
            start, end = window
            logger.info(f"Slicing traces to window: {start}-{end}")
            self.X = np.array(X_full[:, start:end], dtype=np.float32)
        else:
            self.X = np.array(X_full, dtype=np.float32)
            
        # Z-Score Normalization (Per Trace or Global?)
        # Standard approach: Standarize features (per time sample)?
        # Or Normalize trace (per trace)?
        # User said "Normalization: Z-Score is mandatory". Usually per trace.
        # Let's do Standard Scaler per sample (axis 0) if it's consistent hardware.
        # Or per trace (axis 1) if signal level varies.
        # feature_eng.py has normalize_traces using axis=1.
        # We assume X_features is ALREADY normalized/processed?
        # Check feature_eng.py: it calls `normalize_traces` before saving?
        # No, `perform_feature_extraction` saves `X.npy`.
        # I should check if it normalizes.
        
        # Load Y
        self.Y = np.load(y_path).astype(np.longlong)
        
        self.len = len(self.Y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.Y[idx]).long()
