
import os
import argparse
import pandas as pd
import numpy as np
import sys

def check_variance(file_path):
    print(f"Checking: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found.")
        return False
        
    try:
        # Load Metadata
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Check for keys
            # Check for ATC
        elif file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            # NPZ might contain 'metadata' key or just arrays
            if 'metadata' in data:
                 # If stored as df
                 pass 
            # Often handled by ingest logic which is complex.
            # Simplified check: Look for corresponding CSV if it's an NPZ trace file
            csv_path = file_path.replace('.npz', '.csv')
            if os.path.exists(csv_path):
                print(f"Found companion CSV: {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                # Try loading from npz keys
                # key names?
                print("No CSV found. Checking NPZ keys...")
                keys = list(data.keys())
                print(f"Keys: {keys}")
                # This is hard to generalize without ingest logic.
                print("WARNING: Cannot easily check variance on raw NPZ without CSV.")
                return False
        else:
            print("Unsupported format.")
            return False
            
        # Check ATC Variance
        atc_cols = [c for c in df.columns if 'ATC' in c]
        if not atc_cols:
            print("FAIL: No 'ATC' columns found in metadata.")
            # Check for AC_0...AC_7?
            return False
            
        print(f"Found ATC columns: {atc_cols}")
        
        # Check variance
        if len(atc_cols) == 1:
            # String column likely
            vals = df[atc_cols[0]].astype(str).unique()
            n_unique = len(vals)
        else:
            # Multiple columns
            # Aggregate
            sub = df[atc_cols]
            n_unique = sub.drop_duplicates().shape[0]
            
        print(f"Unique Input Values (ATC): {n_unique} / {len(df)}")
        
        if n_unique < 2:
            print("❌ FAIL: Constant Input detected. Cannot be used for Side-Channel Training.")
            return False
        elif n_unique < 100:
            print("⚠️  WARNING: Low variance. Training may be poor.")
            return True
        else:
            print("✅ PASS: Sufficient Input Variance detected.")
            return True
            
    except Exception as e:
        print(f"Error checking file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_variance(sys.argv[1])
    else:
        # Check default dir
        input_dir = "Input/Mastercard"
        print(f"Scanning {input_dir}...")
        for f in os.listdir(input_dir):
            if f.endswith('.csv'):
                check_variance(os.path.join(input_dir, f))
                print("-" * 30)
