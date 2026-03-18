#!/usr/bin/env python
import pandas as pd
import os

# Path to the label file
label_file = "KALKi TEST CARD.xlsx"

if os.path.exists(label_file):
    print(f"Found label file: {label_file}")
    df = pd.read_excel(label_file)
    print(f"\nCurrent contents:")
    print(df)
    
    # Update with correct keys
    correct_keys = {
        'KENC': '9E15204313F7318ACB79B90BD986AD29',
        'KMAC': '4664942FE615FB02E5D57F292AA2B3B6',
        'KDEK': 'CE293B8CC12A977379EF256D76109492'
    }
    
    print(f"\n\nUpdating with correct keys:")
    print(f"  KENC: {correct_keys['KENC']}")
    print(f"  KMAC: {correct_keys['KMAC']}")
    print(f"  KDEK: {correct_keys['KDEK']}")
    
    # Check what columns exist
    print(f"\n\nColumns in file: {list(df.columns)}")
    
    # Update columns based on what exists
    if 'KENC' in df.columns:
        df['KENC'] = correct_keys['KENC']
    elif 'kenc' in df.columns:
        df['kenc'] = correct_keys['KENC']
        
    if 'KMAC' in df.columns:
        df['KMAC'] = correct_keys['KMAC']
    elif 'kmac' in df.columns:
        df['kmac'] = correct_keys['KMAC']
        
    if 'KDEK' in df.columns:
        df['KDEK'] = correct_keys['KDEK']
    elif 'kdek' in df.columns:
        df['kdek'] = correct_keys['KDEK']
    
    # Save back
    df.to_excel(label_file, index=False)
    print(f"\n\nUpdated file saved!")
    print("\nNew contents:")
    print(df)
else:
    print(f"Label file not found: {label_file}")
    print("\nSearching for XLSX files...")
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.xlsx'):
                filepath = os.path.join(root, f)
                print(f"  Found: {filepath}")
