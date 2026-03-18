#!/usr/bin/env python3
"""
Retrain 3DES models with 100% key coverage from external label map
This script addresses the issue where previous models were trained on only 43.9% valid labels
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, "pipeline-code")

def print_header(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Retrain 3DES ensemble models with 100% key coverage"
    )
    parser.add_argument(
        "--label_map",
        default="KALKI TEST CARD.xlsx",
        help="Path to external label map file (XLSX)"
    )
    parser.add_argument(
        "--input_dir",
        default="3des-pipeline/Input",
        help="Input traces directory"
    )
    parser.add_argument(
        "--output_dir",
        default="3des-pipeline/models/3des",
        help="Output models directory"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup old models before retraining"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what will be done without executing"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    
    print_header("3DES MODEL RETRAINING WITH 100% KEY COVERAGE")
    
    print(f"""
This retraining addresses the critical issue discovered in your pipeline:

PROBLEM:
  - Models trained on only 43.9% valid keys (20k/45.6k traces)
  - Remaining 56.1% had missing/invalid keys (label=-1)
  - Models learned from corrupted training signal

SOLUTION:
  - Use external label map (KALKI TEST CARD.xlsx)
  - This provides 100% key coverage (all 45,606 traces have valid keys)
  - Retraining will use complete, valid training data

CONFIGURATION:
  Input directory:     {args.input_dir}
  Label map file:      {args.label_map}
  Output models dir:   {args.output_dir}
  Epochs:              {args.epochs}
  Batch size:          {args.batch_size}
  Backup old models:   {args.backup}
  Dry run:             {args.dry_run}
""")
    
    # Verify files exist
    print_header("STEP 1: VERIFICATION")
    
    input_path = Path(args.input_dir)
    label_path = Path(args.label_map)
    output_path = Path(args.output_dir)
    
    print(f"Checking input directory: {input_path}")
    if input_path.exists():
        trace_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.npz"))
        print(f"  ✓ Found {len(trace_files)} trace files")
    else:
        print(f"  ✗ Input directory not found!")
        return 1
    
    print(f"\nChecking label map: {label_path}")
    if label_path.exists():
        print(f"  ✓ Found {label_path.name}")
    else:
        print(f"  ✗ Label map not found: {label_path}")
        return 1
    
    print(f"\nChecking output models directory: {output_path}")
    if output_path.exists():
        model_files = list(output_path.glob("*.npy")) + list(output_path.glob("*.pth"))
        print(f"  ✓ Found {len(model_files)} existing model files")
        
        if args.backup and not args.dry_run:
            print(f"\n  Creating backup before retraining...")
            backup_path = Path(f"{output_path}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copytree(output_path, backup_path)
            print(f"  ✓ Backup created: {backup_path}")
    else:
        print(f"  ✓ Output directory will be created")
    
    # Load and verify external labels
    print_header("STEP 2: LOAD EXTERNAL LABELS")
    
    if args.dry_run:
        print("DRY RUN: Would load external label map")
    else:
        try:
            from src.external_label_map import load_external_3des_label_map
            
            print(f"Loading external label map from {label_path}...")
            external_labels = load_external_3des_label_map(str(label_path))
            print(f"  ✓ Loaded {len(external_labels)} label entries")
            
            # Show sample
            print(f"\n  Sample entries:")
            for i, (key, val) in enumerate(list(external_labels.items())[:3]):
                print(f"    {key} → {val}")
            
        except Exception as e:
            print(f"  ✗ Error loading external labels: {e}")
            return 1
    
    # Preprocessing step
    print_header("STEP 3: PREPROCESSING WITH 100% KEY COVERAGE")
    
    preprocess_cmd = f"""
python pipeline-code/main.py \\
  --action preprocess \\
  --enable_external_labels \\
  --label_map_xlsx "{label_path}" \\
  --input_dir "{input_path}" \\
  --output_dir "3des-pipeline/Processed/3des"
"""
    
    if args.dry_run:
        print("DRY RUN: Would execute:")
        print(preprocess_cmd)
    else:
        print("Executing preprocessing with external labels...")
        print(f"Command: {preprocess_cmd.strip()}")
        ret = os.system(preprocess_cmd.strip())
        if ret != 0:
            print("  ✗ Preprocessing failed!")
            return 1
        print("  ✓ Preprocessing complete")
    
    # Training step
    print_header("STEP 4: MODEL TRAINING")
    
    train_cmd = f"""
python pipeline-code/main.py \\
  --action train \\
  --epochs {args.epochs} \\
  --batch_size {args.batch_size} \\
  --processed_dir "3des-pipeline/Processed/3des" \\
  --output_dir "3des-pipeline/models/3des"
"""
    
    if args.dry_run:
        print("DRY RUN: Would execute:")
        print(train_cmd)
    else:
        print("Executing training with 100% valid labels...")
        print(f"Command: {train_cmd.strip()}")
        print("\n  This may take 30-60 minutes depending on your hardware...")
        ret = os.system(train_cmd.strip())
        if ret != 0:
            print("  ✗ Training failed!")
            return 1
        print("  ✓ Training complete")
    
    # Validation step
    print_header("STEP 5: VALIDATION")
    
    validation_cmd = """
python pipeline-code/main.py \\
  --action attack \\
  --attack_type inference_3des \\
  --test_data_dir "3des-pipeline/Processed/3des/test" \\
  --models_dir "3des-pipeline/models/3des"
"""
    
    if args.dry_run:
        print("DRY RUN: Would execute:")
        print(validation_cmd)
    else:
        print("Running validation attack on test data...")
        print(f"Command: {validation_cmd.strip()}")
        ret = os.system(validation_cmd.strip())
        if ret != 0:
            print("  ✗ Validation attack failed!")
            return 1
        print("  ✓ Validation complete")
    
    # Summary
    print_header("RETRAINING COMPLETE")
    
    print("""
NEXT STEPS:

1. CROSS-CARD GENERALIZATION TEST (CRITICAL):
   - Current models trained on both Mastercard + Visa from KALKI file
   - To verify generalization:
     a) Split data: Mastercard (90k) vs Visa (50k)
     b) Train on Mastercard only
     c) Test blind on Visa (no labels)
     d) If accuracy > 90%, true generalization achieved
     e) If < 85%, models are memorizing card types

2. PRODUCTION DEPLOYMENT:
   - Only deploy after confirming generalization (step 1)
   - Add confidence scoring to inference output
   - Create monitoring dashboard for key recovery accuracy

3. FUTURE IMPROVEMENTS:
   - Expand KALKI file with 5-10 different keys per card type
   - Retrain to force key-AGNOSTIC pattern learning
   - This eliminates card-type bias in models

4. DOCUMENTATION:
   - Update README with retrain instructions
   - Document model limitations (card-type specific)
   - Note: Models work on blind traces but need labeled traces for validation
""")
    
    return 0

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available for timestamp in backup)")
        import time
        pd = type('pd', (), {'Timestamp': type('Timestamp', (), {
            'now': lambda: type('obj', (), {
                'strftime': lambda fmt: time.strftime(fmt)
            })()
        })()})()
    
    sys.exit(main())
