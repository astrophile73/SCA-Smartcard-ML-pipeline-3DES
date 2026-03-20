#!/usr/bin/env python3
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

npz_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz') and 'rsa' not in f]

print(f"Found {len(npz_files)} NPZ files:")
for fname in npz_files[:2]:
    fpath = os.path.join(npz_dir, fname)
    print(f"\nTesting: {fname}")
    try:
        # Try loading with mmap
        data = np.load(fpath, mmap_mode='r', allow_pickle=False)
        print(f"  Opened successfully with mmap")
        print(f"  Keys: {list(data.keys())}")
        
        if 'trace_data' in data:
            trace_array = data['trace_data']
            print(f"  trace_data shape: {trace_array.shape}, dtype: {trace_array.dtype}")
            print(f"  trace_data is memmap: {isinstance(trace_array, np.memmap)}")
            
            # Try to read a small slice
            print(f"  Attempting to read first 10 traces...")
            batch = np.array(trace_array[:10], dtype=np.float32)
            print(f"  ✓ Read batch successfully: shape={batch.shape}")
        else:
            print(f"  ERROR: No 'trace_data' key in NPZ")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
