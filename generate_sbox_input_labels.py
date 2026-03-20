"""
Week 1 Implementation - Step 1: Generate S-Box Input Labels

This script generates the corrected training labels by extracting 
S-Box INPUT values (6-bit, 0-63) instead of S-Box OUTPUT values (4-bit, 0-15).

Usage:
python generate_sbox_input_labels.py
"""

import sys
from pathlib import Path
import numpy as np

# Add pipeline-code to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline-code"))

from src.gen_labels_sbox_input import generate_sbox_input_labels


def main():
    print("\n" + "="*80)
    print("WEEK 1 STEP 1: Generate S-Box Input Labels")
    print("="*80)
    
    # Paths
    meta_path = Path("3des-pipeline/Processed/3des/Y_meta.csv")
    output_dir = Path("3des-pipeline/Processed/3des")
    
    if not meta_path.exists():
        print(f"❌ Metadata file not found: {meta_path}")
        return 1
    
    print(f"\n[OK] Metadata: {meta_path}")
    print(f"[OK] Output: {output_dir}\n")
    
    # Generate labels for each S-Box
    for sbox_idx in range(8):
        print(f"\nGenerating S-Box {sbox_idx + 1} input labels...")
        try:
            label_file = generate_sbox_input_labels(
                str(meta_path),
                sbox_idx,
                output_dir=str(output_dir),
                key_col="T_DES_KENC",
                stage=1
            )
            print(f"[OK] Saved to: {label_file}")
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    print("\n" + "="*80)
    print("[OK] All S-Box input labels generated successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run: python pipeline-code/src/train_week1.py")
    print("2. This will retrain the model with 64-class output (for 6-bit S-Box inputs)")
    print("3. Expected: Model predicts different keys per trace, ~70-80% per-byte accuracy")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
