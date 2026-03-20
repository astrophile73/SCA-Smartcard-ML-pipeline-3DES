# CRITICAL FINDINGS: Why Model Accuracy is 58%

## Issue #1: Model Predicting SAME Key for All Traces ❌

**Evidence**:
```
Test Report Analysis:
  3DES_KENC: 1 UNIQUE value across 1000 predictions
  3DES_KMAC: 1 UNIQUE value across 1000 predictions  
  3DES_KDEK: 1 UNIQUE value across 1000 predictions

All 1000 samples predicted:
  3DES_KENC = '8007321A40400DC2980107A2C801D008'
  3DES_KMAC = '0186A2670BD026D00210044A5B100185'
  3DES_KDEK = '018980761540251380800B208004510B'
```

**Root Cause**: Model is predicting the **mode (most frequent key)** from training set, not learning key-specific patterns

**Impact**: 
- ✗ The 58% figure is misleading
- ✗ Model isn't doing CPA at all
- ✗ Predictions are essentially random (uniform 1/N chance)

---

## Issue #2: Data Pipeline Architecture Flaw ⚠

The current pipeline structure suggests byte-level models:
```
3des-pipeline/models/3des/
  ├── kdek/      # KDEK (key encryption key) byte models
  ├── kenc/      # KENC (encryption key) byte models
  └── kmac/      # KMAC (MAC key) byte models
```

**But output is**: End-to-End key predictions (combining 24 bytes)

**Problem**: 
- Training is done per-byte (discrete S-Box prediction)
- Inference tries to combine into full 24-byte keys
- Combination method is probably naive (majority voting or averaging)
- This creates a bottleneck where all traces -> same combined key

---

## Issue #3: Processed Data Quality 🔍

**Current Status**:
```
✓ Files: 75 .npy files (already preprocessed)
✓ Traces: 10,000 samples per file
✓ Features: 200 (reduced from ~1000 original)
✓ Data: Already normalized (mean≈0.6, std≈1.0)
```

**Question**: 
- Where is the **training/test split**?
- Are the 10,000 traces per file:
  - A single key? (Would explain why model predicts one key)
  - Multiple keys? (If yes, why isn't model learning to distinguish?)
  - All from same transaction type? (Would explain bias)

---

## Hypothesis: Why Model Always Predicts Same Key

### Scenario A: File-level Key (MOST LIKELY ❌)
```
File structure:
  X_features_s1.npy      → 10,000 traces for 3DES_KENC
  X_features_s2.npy      → 10,000 traces for 3DES_KMAC
  y_labels_s1.npy        → All labels = "3DES_KENC"
  
During training:
  Model learns: feature patterns → key IDENTITY
  Not: feature patterns → key BYTES
```

**Evidence**: Model predicts same KENC for all samples

### Scenario B: Insufficient Key Diversity in Training
```
Training set has:
  - Mastercard key 1: 8000 traces
  - Mastercard key 2: 2000 traces
  
Model learns: Most likely class = Key 1
Result: Predicts Key 1 for everything
```

### Scenario C: Loss Function / Training Issue
```
- No class weighting (imbalanced classes)
- Wrong optimization objective
- Model converged to predicting mode
```

---

## Required Immediate Fixes

### Fix #1: Verify Training Data Structure

```python
# Check what data structure the model actually sees
import numpy as np
from pathlib import Path

# Load sample training files
X = np.load('3des-pipeline/Processed/3des/X_features.npy')  # (10000, 200)
y = np.load('3des-pipeline/Processed/3des/y_labels.npy')     # (10000, 24) or similar?

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Label values (first byte): {np.unique(y[:, 0])}")
print(f"Label distribution (first byte):")
print(pd.Series(y[:, 0]).value_counts().head(10))

# KEY QUESTION: Are these 10,000 samples:
# A) All same key? (10,000xKEY_A) → BAD - Model can only predict KEY_A
# B) Mixed keys? (5000xKEY_A, 5000xKEY_B) → Check for LEARNING
# C) Per-byte classification? (256 classes per byte) → CHECK
```

### Fix #2: Verify Test Procedure

```python
# How are final predictions generated?
# Current seems to be:
#   Per-byte predictions → Majority voting → Final key

# But if per-byte models only see 1-2 keys:
#   Model learns: Always predict KENC_byte_0 = 0x80
#   Result: Final key always 0x80xxxxxxx...
```

### Fix #3: Check Inference Code

```
Need to review: pipeline-code/src/inference_*.py
Questions:
1. How are per-byte predictions combined?
2. Are probability distributions used or just argmax?
3. Is there voting/confidence thresholding?
4. What happens if P(class) < 0.5 for all classes?
```

---

## Immediate Action Plan

### TODAY - Diagnosis
```bash
# 1. Check data structure
python -c "
import numpy as np
y = np.load('3des-pipeline/Processed/3des/y_labels.npy')
print(f'Shape: {y.shape}')
print(f'Dtype: {y.dtype}')
print(f'Range: {y.min()} to {y.max()}')
print(f'Value counts (byte 0): {np.unique(y[:, 0], return_counts=True)}')
"

# 2. Check model predictions (get per-byte probs, not final key)
python pipeline-code/src/inference_fixed.py --debug

# 3. Check training distribution
ls -la 3des-pipeline/Processed/3des/ | grep -E "(X_|y_)"
```

### THIS WEEK - Fix Root Cause
```
Goal: Make model predict DIFFERENT keys for different traces

Options:
A) Generate more diverse training data (different keys)
   - Current: ~3 keys (KENC, KMAC, KDEK) per mastercard
   - Target: Multiple mastercard profiles
   
B) Change training objective
   - Current: Predict key IDENTITY (0=KENC, 1=KMAC, 2=KDEK)
   - Target: Predict key BYTES (0-255 per byte)
   
C) Fix per-byte models
   - Current: Treating each byte independently
   - Target: Learn key-distinguishing patterns per byte
```

### NEXT STEP - Post-Fix Validation
```
After fixes, test should show:
✓ Different predictions for different traces
✓ Predictions vary across all 1000 test samples
✓ Accuracy measured by byte-wise or full-key match
✓ Per-byte accuracy 80%+ or full-key accuracy 40%+
```

---

## Why 58% is Misleading

If model always predicts:
- Same key: Accuracy = P(ground_truth = predicted_key)
- If 1000 test samples have diverse keys: 1/N or less
- If test set is biased (mostly one key): Could be 58%+

**Conclusion**: Current "58%" accuracy is likely:
- On a biased test set
- Or measuring something other than byte/key accuracy
- Not meaningful for CPA attack assessment

---

## Next Investigation Steps

1. **Load test data**: Check if 1000 test samples have 1 key or 1000 keys
2. **Check ground truth**: What are the actual expected keys?
3. **Per-byte breakdown**: accuracy per byte position (not full key)
4. **Model calibration**: Are confidence scores meaningful?

**Expected outcome of fixes**: 
- Phase 1 (Fix data): 58% → ~65-70%
- Phase 2 (Augmentation): 70% → ~80%
- Phase 3 (Architecture): 80% → 85%+

