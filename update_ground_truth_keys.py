#!/usr/bin/env python3
"""Update external label file with correct 3DES keys"""
import pandas as pd
import sys
import os

# Correct keys from user
CORRECT_KEYS = {
    'KENC': '9E15204313F7318ACB79B90BD986AD29',
    'KMAC': '4664942FE615FB02E5D57F292AA2B3B6',
    'KDEK': 'CE293B8CC12A977379EF256D76109492'
}

# Try different possible paths
possible_paths = [
    "KALKi TEST CARD.xlsx",
    "C:/Users/raj50/Desktop/KALKi TEST CARD.xlsx",
    "I:/freelance/SCA Smartcard ML Pipeline-3des/KALKi TEST CARD.xlsx",
    os.path.expanduser("~/KALKi TEST CARD.xlsx"),
]

label_file = None

# Find the file
for path in possible_paths:
    if os.path.exists(path):
        label_file = path
        break

if not label_file:
    print("ERROR: Label file not found at any expected location!")
    print("Searched paths:")
    for p in possible_paths:
        print(f"  {p}")
    sys.exit(1)

print(f"Found label file: {label_file}\n")

# Read current file
df = pd.read_excel(label_file)
print("Current data:")
print(df)
print(f"\nColumns: {list(df.columns)}")

# Update keys
cols_updated = 0
for col_name in df.columns:
    col_lower = col_name.lower()
    for key in CORRECT_KEYS:
        if col_lower == key.lower():
            print(f"\nUpdating column '{col_name}' with {key}: {CORRECT_KEYS[key]}")
            df[col_name] = CORRECT_KEYS[key]
            cols_updated += 1
            break

# Save updated file
df.to_excel(label_file, index=False, engine='openpyxl')
print(f"\n✓ File updated and saved!")
print(f"✓ {cols_updated} columns updated")
print("\nUpdated data:")
print(df)
