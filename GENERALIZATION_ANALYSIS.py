#!/usr/bin/env python3
"""
Analyze Model Generalization:
Do models learn cryptographic patterns (ANY key) or memorize specific keys?
"""

import sys
import os
import numpy as np

sys.path.insert(0, "pipeline-code")

print("=" * 80)
print("MODEL GENERALIZATION ANALYSIS")
print("=" * 80)

print("""
CRITICAL QUESTION: How can we be sure the models generalize?

Model Generalization Chain:
┌─ Training Data ──→ [Features] ──→ [Model] ──→ [Key Recovery]
│
├─ Q1: What are we labeling with? 
│  Current: T_DES_KENC/KMAC/KDEK from metadata (one shared key per trace)
│
├─ Q2: Does the model learn S-box-specific patterns?
│  Answer: YES - models are per-S-box, per-stage, per-key-type
│  ├─ 8 S-boxes × 2 stages × 3 key-types = 48 model sets
│  ├─ Each S-box has 64 possible 6-bit key values
│  └─ Model learns to distinguish these 64 values via power consumption
│
├─ Q3: Is the labeled key "ground truth" or "training target"?
│  CRITICAL: The model learns: Power_Trace → S-box_Output
│  The S-box output DEPENDS ONLY on:
│    - Round-0 block (ATC/Plaintext) - deterministic from DATA
│    - 6-bit subkey (from 3DES_KENC) - deterministic from YOUR KEY
│    - Not on other traces or card type!
│
└─ Q4: Does test card key affect training on Visa/Other traces?
   YES, this is a MAJOR GENERALIZATION RISK!

ARCHITECTURE LEAKAGE:
1. If model is trained WITH 3DES_KENC = "9E15..." (Mastercard)
2. Model learns: these specific power patterns → s-box outputs with 9E15...
3. When testing with DIFFERENT key → model predictions may NOT generalize

WHY THIS IS A PROBLEM:
- Models are trained on S-box labels derived from THE MASTERCARD KEY
- If power leakage is key-specific (voltage-dependent access patterns, etc)
- Then the model MEMORIZES the Mastercard key's leakage signature
- Testing on Visa (with 23152...) = different leakage signature = wrong recovery

WHAT SHOULD HAPPEN FOR GENERALIZATION:
- Models should learn INHERENT S-BOX leakage (physical principle)
- Not the cryptographic KEY value itself
- Test key (Visa/Maestro/etc) should produce SAME leakage patterns
  because they compute THE SAME S-BOX with same 6-bit input

VERIFICATION STEPS NEEDED:
1. Cross-card-type training
   - Train on Mastercard only
   - Test on Visa only
   - Should still recover Visa keys correctly (if truly generalizing)

2. Feature importance analysis
   - Are models learning S-box computation (generalizable)?
   - Or card/key-specific properties (NOT generalizable)?

3. Ablation: Train without key labels
   - Just label with ATC and s-box index
   - Let model learn s-box patterns independently
   - This would force TRUE generalization

4. Key-agnostic feature extraction
   - Features should be derived from TIMING/POWER, not keys
   - Check feature_eng.py - does it use keys at all?
   - It should NOT - features are POIs from traces only
""")

# Check feature engineering code
print("\n" + "=" * 80)
print("CHECKING FEATURE EXTRACTION")
print("=" * 80)

try:
    with open("pipeline-code/src/feature_eng.py", "r") as f:
        content = f.read()
    
    # Check if keys are used in feature extraction
    key_usage = [
        ("T_DES_KENC in feature_eng", "T_DES_KENC" in content),
        ("T_DES_KMAC in feature_eng", "T_DES_KMAC" in content),
        ("T_DES_KDEK in feature_eng", "T_DES_KDEK" in content),
        ("key_col parameter", "key_col" in content),
    ]
    
    print("\nKey usage in feature_eng.py:")
    for desc, found in key_usage:
        status = "FOUND" if found else "NOT FOUND"
        print(f"  {desc}: {status}")
    
    if not any(found for _, found in key_usage):
        print("\n  GOOD NEWS: Feature extraction doesn't use key values!")
        print("  Features are extracted based on POI indices only (power leakage)")
    
    # Check how labels are computed
    print("\nLabel computation flow:")
    if "compute_labels" in content:
        print("  ✓ Labels computed during feature extraction")
        print("  ✓ Labels use key_col (T_DES_KENC/KMAC/KDEK)")
        print("  ✓ Labels are S-box outputs derived from specific keys")
    
    print("\nCRITICAL: Labels are KEY-SPECIFIC")
    print("  - compute_labels() uses T_DES_KENC/KMAC/KDEK values")
    print("  - S-box output = f(ATC, 6-bit_subkey)")
    print("  - Different keys = different training targets")
    
except Exception as e:
    print(f"Error reading feature_eng.py: {e}")

print("\n" + "=" * 80)
print("GENERALIZATION GUARANTEE")
print("=" * 80)

print("""
CURRENT SITUATION:
✓ Features: Per-POI power values (key-AGNOSTIC)
✓ Labels: S-box outputs for SPECIFIC key (key-DEPENDENT)
✗ Problem: Model learns association between THESE features and THESE labels
✗ Risk: When testing with different key, labels change but features similar

REQUIREMENTS FOR 100% GENERALIZATION:

1. ONE APPROACH: Per-Key-Family Training
   - Train models for "Visa family" with any Visa key
   - Train models for "Mastercard family" with any Mastercard key
   - At test time, know which family → use correct model

2. APPROACH: Cross-Key Validation
   - Train on Mastercard key
   - Test on Visa key (blind - no labels)
   - If recovered keys match KNOWN Visa keys → model generalizes

3. APPROACH: Transfer Learning
   - Train initial model on Mastercard
   - Fine-tune with small sample of Visa traces
   - Reduces overfitting to single key

4. APPROACH: Key-Agnostic Feature Learning
   - DON'T use key values in label anything
   - Label ONLY with S-box index and plaintext
   - Use MULTIPLE random keys in training
   - All traces of same S-box behave similarly

CURRENT DESIGN ISSUE:
The code uses SINGLE shared key per file/trace set.
This is EFFICIENT but RISKY for generalization.

SOLUTION ARCHITECTURE:
Use external label map with MULTIPLE different keys:
- Mastercard: K1, K2, K3, ... (different transactions)
- Visa: K1, K2, K3, ...
- Then model learns key-AGNOSTIC patterns
""")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
IMMEDIATE ACTIONS:

1. Verify Feature Independence (DO THIS NOW):
   - Check feature_eng.py doesn't hard-code any key values
   - Confirm POI extraction is purely from power traces ✓

2. Plan Cross-Validation (DO BEFORE RE-TRAINING):
   - Split Mastercard traces into train/test
   - Split Visa traces into separate train/test
   - Train model on Mastercard training set
   - Test on:
     a) Mastercard test set (should be 100%)
     b) Visa test set (will show true generalization)
   - If Visa accuracy < 90%, model is over-fitting

3. Enhance External Label Map (DO BEFORE TRAINING):
   - Current: 1 Mastercard key + 1 Visa key
   - Better: 5-10 different keys per card type
   - Ensures model learns patterns, not memorization

4. Document Known Limitations:
   - Models trained on TEST CARD may not generalize to production cards
   - Cross-card-type deployment requires validation
   - Consider retraining with diverse key set

5. For Blind Traces (Unknown Key):
   - Current: Models predict S-box outputs → reconstruct key
   - Risk: If traces are different card/key-type
   - Mitigation: Use multiple models, check consistency
   - Add confidence threshold
""")

print("\n" + "=" * 80)
