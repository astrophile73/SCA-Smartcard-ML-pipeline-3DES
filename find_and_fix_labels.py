#!/usr/bin/env python
import os
import pandas as pd
from pathlib import Path

# Search for the label file
print("Searching for label files...")
for root, dirs, files in os.walk("I:\\freelance"):
    for f in files:
        if f.endswith('.xlsx') and ('label' in f.lower() or 'kalki' in f.lower() or 'card' in f.lower()):
            filepath = os.path.join(root, f)
            print(f"\nFound: {filepath}")
            try:
                df = pd.read_excel(filepath)
                print(f"  Columns: {list(df.columns)}")
                print(f"  Shape: {df.shape}")
                if len(df) > 0:
                    # Show first few rows
                    print(f"\n  First 3 rows:")
                    print(df.head(3).to_string())
            except Exception as e:
                print(f"  Error reading: {e}")
