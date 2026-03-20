# Accuracy Improvement Roadmap
## 3DES ML Pipeline CPA Attack

**Current State**: ~58% accuracy (significantly below the 85%+ target)
**Root Cause**: Insufficient training data + key leakage model mismatch
**Objective**: Achieve 85%+ accuracy through systematic improvements

---

## Phase 1: Quick Wins (Week 1)
*(No new data required - optimize with existing data)*

### 1.1 Data Analysis & Diagnosis
- [ ] **Check data imbalance**: S-Box input space coverage
  ```python
  # In pipeline-code/src/analyze_rounds.py
  # Add histogram of S-Box inputs by round/byte
  ```
- [ ] **Verify leakage alignment**: Are traces properly time-synchronized?
  - Check if peak power consumption aligns with known S-Box rounds
  - If not: re-synchronize traces before training

### 1.2 Model Architecture Fix
- [ ] **Review MLP topology**: Current may be under-parameterized
  - Try: `128 → 128 → 64` (instead of current)
  - Add dropout: 0.3 between layers
  
- [ ] **Verify loss function**: Should use weighted cross-entropy
  - Weight classes inversely by frequency
  - Prevents model learning only common S-Box inputs

### 1.3 Training Hyperparameter Tuning
- [ ] **Batch size test**: Try [16, 32, 64, 128]
- [ ] **Learning rate schedule**: Use exponential decay
  - Start: 0.001 → End: 0.0001 (over 50 epochs)
- [ ] **Early stopping**: Monitor validation accuracy, patience=10

**Expected Improvement**: 58% → ~65-70%

---

## Phase 2: Data Augmentation (Week 2)
*(Multiply training data 3-5x without collecting more traces)*

### 2.1 Trace Augmentation Techniques

#### Technique 1: Gaussian Noise Injection
```python
# Add realistic measurement noise
noisy_traces = traces + np.random.normal(0, σ, traces.shape)
# σ depends on your ADC noise floor (typically 0.5-2% of signal amplitude)
```

#### Technique 2: Time-Series Jitter
```python
# Simulate timing variations (±1-3 samples)
shifted = np.roll(traces, np.random.randint(-3, 4), axis=1)
```

#### Technique 3: Amplitude Scaling
```python
# Simulate different probe distances/coupling
scaled = traces * np.random.uniform(0.95, 1.05, (n_traces, 1))
```

#### Technique 4: Hamming Weight Variations
```python
# For byte-level leakage, add variations based on HW model
hw = bin(sbox_input).count('1')
noise = np.random.normal(hw * base_coeff, std_dev, traces.shape)
```

### 2.2 Implementation Steps

1. **Load real Mastercard traces** (from `3des-pipeline/Input/`)
2. **Apply augmentation**: Generate 3-5 variations per trace
3. **Create balanced dataset**: Ensure equal samples per S-Box input (0-255)
4. **Retrain models** with augmented data

**Augmentation Script**:
```python
# Create augmented_datasets/
for dataset_name in ['mastercard', 'visa']:
    original = load_traces(f"3des-pipeline/Input/{dataset_name}*.csv")
    
    augmented = []
    for trace in original:
        augmented.append(trace)  # Original
        augmented.append(add_noise(trace, sigma=0.01))
        augmented.append(time_shift(trace, max_shift=3))
        augmented.append(scale_amplitude(trace, range=(0.95, 1.05)))
        augmented.append(apply_hamming_weight_model(trace))
    
    # Save with balanced class distribution per byte
    save_balanced_dataset(augmented, f"augmented_datasets/{dataset_name}/")
```

**Expected Improvement**: 70% → ~75-80%

---

## Phase 3: Multi-byte Learning (Week 3)
*(Leverage interdependencies between key bytes)*

### 3.1 Current Approach Issues
- **Problem**: Treating each byte independently
- **Reality**: 3DES key bytes interact through the algorithm

### 3.2 Improved Approach (Ensemble)

1. **Byte-level models** (CURRENT): Predicts each byte independently
2. **Relationship models**: Learn byte-to-byte dependencies
3. **Joint inference**: Combine predictions considering dependencies

**Implementation**:
```python
# Train relationship models between adjacent key bytes
for byte_idx in range(24):  # 3DES = 24 bytes
    for next_byte in [byte_idx-1, byte_idx+1]:
        # Model how key[byte_idx] correlates with key[next_byte]
        # Input: (S-Box input for byte_idx, predicted byte_idx)
        # Output: probability distribution for key[next_byte]
```

**Expected Improvement**: 80% → ~85%+

---

## Phase 4: Test Set Validation (Week 4)

### 4.1 Before Deployment
- [ ] Test on **held-out dataset** (20% of Mastercard data)
- [ ] **Cross-validation**: 5-fold on augmented data
- [ ] **Confidence metrics**: Track top-1 vs top-5 accuracy

### 4.2 Key Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Per-byte accuracy | 95%+ | 58% |
| Full 24-byte match | 85%+ | 0% |
| Top-5 accuracy | 99%+ | ? |

---

## Detailed Improvement Map

### Quick Diagnostic (DO TODAY)
```python
# pipeline-code/src/diagnose.py
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Check data variance by byte
print("=== Data Variance Analysis ===")
for byte_idx in range(24):
    traces_by_input = {}
    for sbox_input in range(256):
        mask = (labels[:, byte_idx] == sbox_input)
        traces_by_input[sbox_input] = traces[mask].var(axis=0).mean()
    
    coverage = sum(1 for v in traces_by_input.values() if v > 0)
    print(f"Byte {byte_idx}: {coverage}/256 inputs seen")

# 2. Check trace quality
print("\n=== Trace Quality ===")
print(f"Mean SNR: {traces.std() / traces[:, ::10].std():.2f}")

# 3. Check label correctness
print("\n=== Label Verification ===")
expected_keys = {
    'T_DES_KENC': '9E15204313F7318ACB79B90BD986AD29',
    'T_DES_KMAC': '4664942FE615FB02E5D57F292AA2B3B6',
}
# Verify label extraction matches expected keys
```

### Week 1 Checklist
- [ ] Run diagnostic
- [ ] Fix architecture (update `pipeline-code/src/model.py`)
- [ ] Add weighted loss (update training code)
- [ ] Retrain with better hyperparams
- [ ] Measure improvement

### Week 2 Checklist
- [ ] Create augmentation module
- [ ] Generate augmented dataset (3x multiplier)
- [ ] Retrain with augmented data
- [ ] Validate on original test set
- [ ] Measure improvement

### Week 3 Checklist
- [ ] Analyze byte correlations
- [ ] Train byte relationship models
- [ ] Implement ensemble inference
- [ ] Test combined approach
- [ ] Measure final accuracy

### Week 4 Checklist
- [ ] Cross-validate all changes
- [ ] Test on held-out Visa data
- [ ] Generate final accuracy report
- [ ] Document improvement factors

---

## Dataset Status

### Mastercard Dataset
- **Location**: `3des-pipeline/Input/mastercard*.csv`
- **Traces**: Check with: `python -c "import pandas as pd; print(sum(len(pd.read_csv(f)) for f in Path('3des-pipeline/Input').glob('mastercard*.csv')))"`
- **Ground Truth Keys**: Embedded in CSV
- **Augmentation Strategy**: Noise injection + time shift + amplitude scaling

### Visa Dataset (IF AVAILABLE)
- **Location**: `3des-pipeline/Input/visa*.csv`
- **Strategy**: Use if traces are time-aligned with Mastercard

---

## Success Criteria

| Milestone | Target | Timeline |
|-----------|--------|----------|
| Phase 1 (Optimization) | 70% | Week 1 |
| Phase 2 (Augmentation) | 80% | Week 2 |
| Phase 3 (Ensemble) | 85%+ | Week 3 |
| Validation | 85%+ on test set | Week 4 |

---

## Files to Modify

1. **pipeline-code/src/model.py** - Architecture & loss function
2. **pipeline-code/src/train.py** - Hyperparameters & early stopping
3. **pipeline-code/src/data_augmentation.py** (NEW) - Augmentation techniques
4. **pipeline-code/src/ensemble_inference.py** - Multi-byte relationships (if needed)

---

## Quick Start Commands

```bash
# Week 1: Diagnostic + Retraining
python pipeline-code/src/diagnose.py
python pipeline-code/main.py --train --epochs=100 --batch-size=32

# Week 2: Data Augmentation
python pipeline-code/src/data_augmentation.py
python pipeline-code/main.py --train --augmented --epochs=150

# Week 3: Ensemble Approach (if needed)
python pipeline-code/train_ensemble.py  # Already exists!

# Week 4: Validation
python pipeline-code/main.py --test --validate --cross-fold=5
```

---

**Next Step**: Start with Phase 1 diagnostics to confirm root causes before proceeding with data augmentation.

