#!/usr/bin/env python
"""
Fix corrupted normalization statistics in 3DES models.
The mean/std files were incorrectly saved with shape (1500,) instead of (38,).
This script recomputes them correctly from the processed features.
"""
import numpy as np
import os
from pathlib import Path

# Load the processed features
processed_dir = "3des-pipeline/Processed/3des"
model_root = "3des-pipeline/models"

print("Loading processed features...")
X_s1 = np.load(os.path.join(processed_dir, "X_features_s1.npy")).astype(np.float32)
X_s2 = np.load(os.path.join(processed_dir, "X_features_s2.npy")).astype(np.float32)

print(f"X_s1 shape: {X_s1.shape}")
print(f"X_s2 shape: {X_s2.shape}")

# Compute correct statistics
mean_s1 = np.mean(X_s1, axis=0)
std_s1 = np.std(X_s1, axis=0)
std_s1[std_s1 == 0] = 1

mean_s2 = np.mean(X_s2, axis=0)
std_s2 = np.std(X_s2, axis=0)
std_s2[std_s2 == 0] = 1

print(f"Computed mean_s1 shape: {mean_s1.shape}")
print(f"Computed std_s1 shape: {std_s1.shape}")
print(f"Computed mean_s2 shape: {mean_s2.shape}")
print(f"Computed std_s2 shape: {std_s2.shape}")

# Save to all key-type directories
key_types = ["kenc", "kmac", "kdek"]
for key_type in key_types:
    kt_dir = os.path.join(model_root, "3des", key_type)
    os.makedirs(kt_dir, exist_ok=True)
    
    mean_s1_path = os.path.join(kt_dir, "mean_s1.npy")
    std_s1_path = os.path.join(kt_dir, "std_s1.npy")
    mean_s2_path = os.path.join(kt_dir, "mean_s2.npy")
    std_s2_path = os.path.join(kt_dir, "std_s2.npy")
    
    print(f"\nSaving for {key_type.upper()}...")
    np.save(mean_s1_path, mean_s1)
    np.save(std_s1_path, std_s1)
    np.save(mean_s2_path, mean_s2)
    np.save(std_s2_path, std_s2)
    print(f"  Saved {mean_s1_path}")
    print(f"  Saved {std_s1_path}")
    print(f"  Saved {mean_s2_path}")
    print(f"  Saved {std_s2_path}")

# Also save at root level for backward compatibility
np.save(os.path.join(model_root, "mean_s1.npy"), mean_s1)
np.save(os.path.join(model_root, "std_s1.npy"), std_s1)
np.save(os.path.join(model_root, "mean_s2.npy"), mean_s2)
np.save(os.path.join(model_root, "std_s2.npy"), std_s2)
print(f"\nAlso saved at model root for backward compatibility")

print("\nDone! Normalization statistics have been fixed.")
