#!/usr/bin/env python3
"""
Debug: Check feature and statistics dimensions
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline-code"))

processed_dir = "3des-pipeline/Processed/3des"
model_dir = "3des-pipeline/models/3des"

print("=" * 70)
print("DIMENSION ANALYSIS")
print("=" * 70)

# Load features
X1_path = os.path.join(processed_dir, "X_features_s1.npy")
X2_path = os.path.join(processed_dir, "X_features_s2.npy")

if os.path.exists(X1_path):
    X1 = np.load(X1_path)
    print(f"\nX_features_s1.npy shape: {X1.shape}")
else:
    print(f"\n✗ X_features_s1.npy not found")

if os.path.exists(X2_path):
    X2 = np.load(X2_path)
    print(f"X_features_s2.npy shape: {X2.shape}")
else:
    print(f"✗ X_features_s2.npy not found")

# Check statistics for each key type
print(f"\nNormalization statistics:")
for key_type in ["kenc", "kmac", "kdek"]:
    for stage in [1, 2]:
        mean_path = os.path.join(model_dir, key_type, f"mean_s{stage}.npy")
        std_path = os.path.join(model_dir, key_type, f"std_s{stage}.npy")
        
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = np.load(mean_path)
            std = np.load(std_path)
            print(f"\n{key_type.upper()} Stage {stage}:")
            print(f"  mean_s{stage}.npy shape: {mean.shape}")
            print(f"  std_s{stage}.npy shape: {std.shape}")
        else:
            print(f"\n{key_type.upper()} Stage {stage}: FILES NOT FOUND")

# Check what features are per-key-type
print(f"\n\nPer-key-type features check:")
for key_type in ["kenc", "kmac", "kdek"]:
    for stage in [1, 2]:
        feature_path = os.path.join(processed_dir, f"X_features_{key_type}_s{stage}.npy")
        if os.path.exists(feature_path):
            features = np.load(feature_path)
            print(f"X_features_{key_type}_s{stage}.npy: {features.shape}")
        else:
            print(f"X_features_{key_type}_s{stage}.npy: NOT FOUND")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print("""
If X_features_s1.npy has shape (10000, 1146) but mean_s1.npy has shape (1178,),
this indicates:
1. Features were extracted differently than statistics were computed, OR
2. Statistics are for a different feature set (perhaps after some post-processing)

Possible solutions:
- Check if statistics should match the feature dimension (1146)
- Or check if features should be reshaped to match statistics (1178)
""")
