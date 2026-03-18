import numpy as np
import os

os.chdir("3des-pipeline/Processed/3des")

files_to_check = [
    'X_features_kenc_s1.npy',
    'X_features_kmac_s1.npy',
    'X_features_kdek_s1.npy',
    'X_features_s1.npy',
    'X_features.npy'
]

for f in files_to_check:
    if os.path.exists(f):
        shape = np.load(f).shape
        print(f"{f}: {shape}")
    else:
        print(f"{f}: NOT FOUND")
