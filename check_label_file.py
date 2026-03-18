#!/usr/bin/env python3
"""
Check the KALKi label file contents
"""

import pandas as pd
import os

label_files = [
    "KALKi TEST CARD.xlsx",
    "I:\freelance\SCA Smartcard ML Pipeline-3des\KALKi TEST CARD.xlsx",
    "I:\freelance\SCA-Smartcard-Pipeline-3\KALKi TEST CARD.xlsx",
]

print("=" * 80)
print("KALKI TEST CARD LABEL FILE ANALYSIS")
print("=" * 80)

for fpath in label_files:
    if os.path.exists(fpath):
        print(f"\nFound: {fpath}\n")
        
        try:
            df = pd.read_excel(fpath)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}\n")
            
            print("Data:")
            print(df.to_string())
            
        except Exception as e:
            print(f"Error reading: {e}")
        break
else:
    print("\nNo file found at expected locations:")
    for fpath in label_files:
        print(f"  {fpath}")

print("\n" + "=" * 80)
