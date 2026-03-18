import os
import numpy as np

input_dir = r'3des-pipeline/Input'

npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])

print("=" * 80)
print("DETAILED NPZ ARRAY ANALYSIS")
print("=" * 80)

for npz_file in npz_files[:3]:  # Check first 3 files
    filepath = os.path.join(input_dir, npz_file)
    print(f"\n{'='*80}")
    print(f"File: {npz_file}")
    print(f"{'='*80}")
    
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"Keys: {list(data.keys())}\n")
        
        for key in data.keys():
            arr = data[key]
            print(f"  Key: '{key}'")
            print(f"    Type: {type(arr)}")
            print(f"    DType: {arr.dtype}")
            print(f"    Shape: {arr.shape}")
            print(f"    Ndim: {arr.ndim}")
            
            # Check if it's a scalar
            if arr.ndim == 0:
                print(f"    >>> SCALAR VALUE: {arr.item()}")
            else:
                print(f"    First element: {arr.flat[0]}")
            print()
        
        data.close()
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}\n")

print("=" * 80)
print("END ANALYSIS")
print("=" * 80)
