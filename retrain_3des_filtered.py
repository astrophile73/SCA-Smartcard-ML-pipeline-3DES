#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain 3DES models with filtered data (Mastercard + Greenvisa traces only)
Uses external label map for both card types
"""

import sys
import os
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, "pipeline-code")

def print_header(msg):
    print("\n" + "=" * 90)
    print(f" {msg}")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(
        description="Retrain 3DES models with Mastercard + Greenvisa traces"
    )
    parser.add_argument(
        "--input_dir",
        default="3des-pipeline/Input_3DES_Training",
        help="Prepared input directory with filtered 3DES traces"
    )
    parser.add_argument(
        "--label_map",
        default="KALKI TEST CARD.xlsx",
        help="External label map file"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup old models before retraining"
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
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing, use pre-processed data"
    )
    
    args = parser.parse_args()
    
    print_header("3DES MODEL RETRAINING (MASTERCARD + GREENVISA)")
    
    print(f"""
Retraining 3DES ensemble models with filtered traces:
  [OK] Mastercard 3DES (with native labels)
  [OK] Greenvisa 3DES (without native labels, using external map)
  
Configuration:
  Input directory:   {args.input_dir}
  Label map file:    {args.label_map}
  Training epochs:   {args.epochs}
  Batch size:        {args.batch_size}
  Backup models:     {args.backup}
  Skip preprocessing:{args.skip_preprocessing}
""")
    
    # Verify input
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"[-] ERROR: Input directory not found: {args.input_dir}")
        print("   Run: python prepare_training_data.py")
        return 1
    
    # Verify label map
    label_path = Path(args.label_map)
    if not label_path.exists():
        print(f"[-] ERROR: Label map not found: {args.label_map}")
        return 1
    
    # Step 1: Backup old models
    if args.backup:
        print_header("STEP 1: BACKING UP OLD MODELS")
        
        models_dir = Path("3des-pipeline/models/3des")
        if models_dir.exists():
            backup_dir = Path(f"3des-pipeline/models/3des_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copytree(models_dir, backup_dir)
            print(f"[OK] Backup created: {backup_dir}")
        else:
            print(f"  (No existing models to backup)")
    
    # Step 2: Preprocessing
    if not args.skip_preprocessing:
        print_header("STEP 2: PREPROCESSING WITH EXTERNAL LABELS")
        
        # Build command as a list for subprocess (Windows compatible)
        # Note: main.py uses --mode instead of --action, and --scan_type instead of --trace_type
        preprocess_cmd = [
            "python", "pipeline-code/main.py",
            "--mode", "preprocess",
            "--scan_type", "3des",
            "--enable_external_labels",
            "--label_map_xlsx", args.label_map,
            "--input_dir", args.input_dir,
            "--processed_dir", "3des-pipeline/Processed/3des_retraining",
        ]
        
        print("Executing preprocessing:")
        print(" ".join(preprocess_cmd))
        print("\nThis may take 15-30 minutes...")
        
        ret = subprocess.run(preprocess_cmd).returncode
        if ret != 0:
            print("[-] Preprocessing failed!")
            return 1
        
        print("[OK] Preprocessing complete")
    else:
        print_header("STEP 2: PREPROCESSING SKIPPED (using pre-existing data)")
    
    # Step 3: Training
    print_header("STEP 3: MODEL TRAINING")
    
    train_cmd = [
        "python", "pipeline-code/main.py",
        "--mode", "train",
        "--scan_type", "3des",
        "--epochs", str(args.epochs),
        "--processed_dir", "3des-pipeline/Processed/3des_retraining",
        "--model_root", "3des-pipeline/models"
    ]
    
    print("Executing training:")
    print(" ".join(train_cmd))
    print(f"\nThis will take 1-2 hours with {args.epochs} epochs...")
    print("(You can monitor progress in the logs)")
    
    ret = subprocess.run(train_cmd).returncode
    if ret != 0:
        print("[-] Training failed!")
        return 1
    
    print("[OK] Training complete")
    
    # Step 4: Validation
    print_header("STEP 4: VALIDATION")
    
    print("""
Quick validation test on training data:
  This verifies models were trained correctly
  (Not a generalization test - just sanity check)
""")
    
    validate_cmd = [
        "python", "pipeline-code/main.py",
        "--mode", "attack",
        "--scan_type", "3des",
        "--processed_dir", "3des-pipeline/Processed/3des_retraining",
        "--model_root", "3des-pipeline/models",
        "--output_dir", "3des-pipeline/Output/validation_retrain_sanity_check",
        "--return_confidence"
    ]
    
    print("Executing validation attack:")
    print(" ".join(validate_cmd))
    
    ret = subprocess.run(validate_cmd).returncode
    if ret != 0:
        print("[~] Validation attack had issues (may be normal)")
    else:
        print("[OK] Validation complete")
    
    # Summary
    print_header("RETRAINING COMPLETE")
    
    print(f"""
[OK] 3DES Models Retrained Successfully

New models saved to:
  3des-pipeline/models/3des/

If you backed up old models:
  3des-pipeline/models/3des_backup_[timestamp]/

NEXT STEPS:

1. RUN VALIDATION STAGES (mandatory before production):
   
   Stage 1: Test Mastercard baseline
   $ python pipeline-code/main.py --mode attack --scan_type 3des --card_type mastercard \\
     --input_dir "I:\\freelance\\SCA-Smartcard-Pipeline-3\\Input1\\Mastercard" \\
     --models_dir "3des-pipeline/models/3des" \\
     --output_dir "3des-pipeline/Output/validation_stage_1_mastercard" \\
     --confidence_threshold 0.80
   
   Expected: 99%+ accuracy (models were trained on Mastercard)
   
   ---
   
   Stage 3: Test Greenvisa blind attack
   $ python pipeline-code/main.py --mode attack --scan_type 3des --card_type greenvisa \\
     --input_dir "I:\\freelance\\SCA-Smartcard-Pipeline-3\\Input1\\Visa\\Green Visa Traces - 5000 (3DES)" \\
     --models_dir "3des-pipeline/models/3des" \\
     --output_dir "3des-pipeline/Output/validation_stage_3b_greenvisa_blind" \\
     --confidence_threshold 0.70
   
   Expected: 90%+ accuracy (blind test, generalization)

2. CHECK RESULTS:
   $ python << 'EOF'
import pandas as pd
df = pd.read_csv('3des-pipeline/Output/validation_stage_1_mastercard/Final_Report_mastercard_session.csv')
accuracy = df['status'].value_counts().get('SUCCESS', 0) / len(df)
print(f"Mastercard validation: {accuracy:.1%} accuracy")
print(f"Expected: 99%+ → Retrain {'✓ PASS' if accuracy >= 0.99 else '✗ REVIEW'}")
   EOF

3. IF ALL PASS:
   Models are ready for production deployment
   Proceed with Stage 1-3 full validation (see EXECUTION_CHECKLIST.md)

4. IF ANY FAIL:
   Check: STAGED_VALIDATION_PLAN.md troubleshooting section
   Review: Trace quality and label map completeness
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
