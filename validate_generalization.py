#!/usr/bin/env python3
"""
Cross-Card Generalization Validation
This tests whether 3DES models trained on one card type (e.g. Mastercard)
can successfully recover keys from a different card type (e.g. Visa)
"""

import sys
import os
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, "pipeline-code")

def print_header(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def filter_traces_by_profile(traces_dir, profile_pattern):
    """
    Filter traces to only those matching a specific card profile/type
    Example: profile_pattern = "Visa" or "Mastercard"
    """
    traces_dir = Path(traces_dir)
    matching_files = []
    
    for f in traces_dir.glob("*.csv"):
        # Check if profile matches in filename or metadata
        if profile_pattern.lower() in f.name.lower():
            matching_files.append(f)
    
    return matching_files

def analyze_trace_profiles(input_dir):
    """
    Analyze available traces and identify unique profiles/card types
    """
    from src.ingest import TraceDataset
    
    print("Analyzing trace profiles...")
    
    input_path = Path(input_dir)
    trace_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.npz"))
    
    profiles = {}
    
    for trace_file in trace_files:
        print(f"  Examining {trace_file.name}...")
        try:
            # Extract profile info from filename or metadata
            name_parts = trace_file.stem.split("_")
            
            # Heuristic: look for Visa/Mastercard/Amex in filename
            if "visa" in trace_file.name.lower():
                profile = "Visa"
            elif "mastercard" in trace_file.name.lower():
                profile = "Mastercard"
            elif "amex" in trace_file.name.lower():
                profile = "Amex"
            else:
                profile = "Unknown"
            
            if profile not in profiles:
                profiles[profile] = []
            profiles[profile].append(trace_file)
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print(f"\nIdentified {len(profiles)} profiles:")
    for profile, files in profiles.items():
        print(f"  {profile}: {len(files)} files")
    
    return profiles

def main():
    parser = argparse.ArgumentParser(
        description="Validate 3DES model generalization across card types"
    )
    parser.add_argument(
        "--input_dir",
        default="3des-pipeline/Input",
        help="Input traces directory"
    )
    parser.add_argument(
        "--processed_dir",
        default="3des-pipeline/Processed/3des",
        help="Processed data directory"
    )
    parser.add_argument(
        "--models_dir",
        default="3des-pipeline/models/3des",
        help="Models directory"
    )
    parser.add_argument(
        "--train_profile",
        default="Mastercard",
        help="Card profile to train on (e.g. Mastercard)"
    )
    parser.add_argument(
        "--test_profile",
        default="Visa",
        help="Card profile to test on (e.g. Visa)"
    )
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=0.90,
        help="Minimum accuracy for passing generalization test (default 0.90)"
    )
    
    args = parser.parse_args()
    
    print_header("CROSS-CARD GENERALIZATION VALIDATION")
    
    print(f"""
This test validates whether the 3DES models truly learn cryptographic patterns
or if they memorize specific card types.

METHODOLOGY:
1. Train ensemble models on {args.train_profile} traces only
2. Test models on {args.test_profile} traces (blind - no labels provided)
3. Compare recovered keys against known {args.test_profile} keys
4. If accuracy > {args.accuracy_threshold*100:.0f}%, generalization is proven

WHY THIS MATTERS:
- If models only learn card-type-specific leakage, they'll fail on Visa
- If models learn true S-box computation patterns, they'll work on any key
- Current KALKI has only 1 Mastercard + 1 Visa = high over-fitting risk
""")
    
    # Step 1: Analyze available data
    print_header("STEP 1: ANALYZE AVAILABLE PROFILES")
    
    try:
        profiles = analyze_trace_profiles(args.input_dir)
        
        if args.train_profile not in profiles:
            print(f"\n✗ ERROR: {args.train_profile} traces not found!")
            print(f"   Available profiles: {list(profiles.keys())}")
            return 1
        
        if args.test_profile not in profiles:
            print(f"\n✗ ERROR: {args.test_profile} traces not found!")
            print(f"   Available profiles: {list(profiles.keys())}")
            return 1
        
        print(f"\n✓ Found {args.train_profile} and {args.test_profile} traces")
        
    except Exception as e:
        print(f"\n✗ Error analyzing profiles: {e}")
        return 1
    
    # Step 2: Split and preprocess data
    print_header("STEP 2: SPLIT DATA BY PROFILE")
    
    print(f"""
Preprocessing step should:
1. Load all {args.train_profile} traces
2. Extract features + labels using KALKI keys
3. Train ensemble on {args.train_profile} data

Then:
1. Load all {args.test_profile} traces  
2. Extract features (NO labels - blind test)
3. Inference using {args.train_profile}-trained models
4. Compare predicted keys vs actual {args.test_profile} keys
""")
    
    # Step 3: Train on one profile
    print_header("STEP 3: TRAIN ON {args.train_profile}")
    
    train_cmd = f"""
python pipeline-code/main.py \\
  --action preprocess \\
  --enable_external_labels \\
  --label_map_xlsx "KALKI TEST CARD.xlsx" \\
  --input_dir "{args.input_dir}" \\
  --profile_filter "{args.train_profile}" \\
  --output_dir "{args.processed_dir}/{args.train_profile}"
"""
    
    print(f"Command: {train_cmd.strip()}\n")
    print("Note: This preprocessing step needs --profile_filter parameter")
    print(f"      to select only {args.train_profile} traces")
    
    # Step 4: Test on different profile
    print_header("STEP 4: TEST ON {args.test_profile} (BLIND)")
    
    test_cmd = f"""
python pipeline-code/main.py \\
  --action attack \\
  --attack_type inference_3des \\
  --input_dir "{args.input_dir}" \\
  --profile_filter "{args.test_profile}" \\
  --models_dir "{args.models_dir}" \\
  --output_report "{args.processed_dir}/generalization_report.csv"
"""
    
    print(f"Command: {test_cmd.strip()}\n")
    print(f"This will:")
    print(f"  1. Load {args.test_profile} traces WITHOUT labels (blind)")
    print(f"  2. Use models trained on {args.train_profile}")
    print(f"  3. Recover keys using ensemble voting")
    print(f"  4. Compare against known {args.test_profile} keys from KALKI")
    
    # Step 5: Interpret results
    print_header("STEP 5: INTERPRET RESULTS")
    
    print(f"""
RESULT INTERPRETATION:

╔══════════════════════════════════════════════════════════════════╗
║ Accuracy > {args.accuracy_threshold*100:.0f}%:  ✓ TRUE GENERALIZATION ACHIEVED        ║
║   → Models learned cryptographic patterns                        ║
║   → Safe to deploy on new card types                            ║
║   → Recommendation: Proceed to production                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Accuracy 70-90%: ⚠ PARTIAL GENERALIZATION                        ║
║   → Models learn SOME card-independent patterns                 ║
║   → Some card-type bias remains                                 ║
║   → Recommendation: Retrain with more diverse keys             ║
╠══════════════════════════════════════════════════════════════════╣
║ Accuracy < 70%: ✗ NO GENERALIZATION                             ║
║   → Models memorized specific keys/card types                   ║
║   → UNABLE to attack unknown cards                             ║
║   → Recommendation: Review archi tecture, expand training data  ║
╚══════════════════════════════════════════════════════════════════╝

DIAGNOSTIC STEPS IF GENERALIZATION FAILS:

1. Check Data Leakage:
   - Confirm training used {args.train_profile} ONLY
   - Confirm test used {args.test_profile} ONLY
   - No mixing of profiles during training

2. Feature Analysis:
   - Are POIs consistent across card types?
   - Do features capture true power leakage?
   - Or do they capture card-specific artifacts?

3. Label Quality:
   - Are KALKI labels accurate for both profiles?
   - Do label computations work for both key values?
   - Verify S-box outputs are correct

4. Model Capacity:
   - Are ensemble models large enough to learn patterns?
   - Are epochs sufficient for convergence?
   - Check training loss convergence

5. Key Diversity:
   - Current: 1 Mastercard + 1 Visa key
   - Better: 5-10 keys per profile
   - Expand KALKI to diversify training
""")
    
    # Step 6: Enhanced training (if needed)
    print_header("STEP 6: IF GENERALIZATION FAILS - ENHANCED TRAINING")
    
    print(f"""
To force key-agnostic learning:

1. EXPAND EXTERNAL LABEL MAP:
   Create expanded_KALKI.xlsx with:
   ┌─ Mastercard Cards
   │  ├─ Card 1 (from KALKI): Key = 9E15204313F...
   │  ├─ Card 2 (if available): Key = XXXXXXXXXXXXXXX
   │  ├─ Card 3 (if available): Key = XXXXXXXXXXXXXXX
   │  └─ ... (aim for 5-10 keys)
   │
   └─ Visa Cards
      ├─ Card 1 (from KALKI): Key = 23152081ECF...
      ├─ Card 2 (if available): Key = XXXXXXXXXXXXXXX
      ├─ Card 3 (if available): Key = XXXXXXXXXXXXXXX
      └─ ... (aim for 5-10 keys)

2. RETRAIN:
   python retrain_with_full_coverage.py \\
     --label_map expanded_KALKI.xlsx \\
     --epochs 300

3. REVALIDATE GENERALIZATION:
   python validate_generalization.py \\
     --accuracy_threshold 0.95
""")
    
    print_header("SUMMARY")
    
    print(f"""
This generalization test is CRITICAL because:

1. Current models may only work with KALKI test cards
2. Production cards may have different leakage signatures
3. Cross-card testing BEFORE deployment prevents field failures

EXPECTED TIMELINE:
  - Step 1-2: < 1 minute
  - Step 3: 30-60 minutes (training)
  - Step 4: 5 minutes (inference)
  - Total: ~45-90 minutes

CONTACT POINTS:
  If generalization PASSES:
    → Models are ready for production
    → Document assumptions (card families trained on)
    → Set up monitoring for key recovery accuracy

  If generalization FAILS:
    → Need to expand KALKI with more keys
    → May need different model architecture
    → Consider per-card-type training
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
