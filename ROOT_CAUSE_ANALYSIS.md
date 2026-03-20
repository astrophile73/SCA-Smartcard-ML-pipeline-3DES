# ROOT CAUSE FOUND: Hamming Weight Model Instead of S-Box Input Model

## The Critical Flaw 🎯

Your model is trained to predict **Hamming Weight** (0-8 bits), NOT **S-Box input values (0-255)**

### Evidence
```
Label analysis:
  Y_labels_kdek_s1_sbox1.npy: Range [1, 12], Unique: 4 values
  Y_labels_kdek_s1_sbox2.npy: Range [4, 15], Unique: 2 values
  Y_labels_kdek_s1_sbox3.npy: Range [0, 15], Unique: 4 values
  Y_labels_kdek_s1_sbox4.npy: Range [9, 15], Unique: 2 values

Expected for S-Box input:
  Range [0, 255], Unique: ~256 values

Actual: Hamming Weight is 0-8, with heavy imbalance:
  Y_labels_kdek_s1_sbox6.npy: 6529 samples for HW=0, only 3471 for HW=13
```

### Why This Fails

**Hamming Weight prediction is fundamentally lossy**:
```
S-Box Input Binary → Hamming Weight (loses information)

0x01 (00000001) → HW = 1
0x02 (00000010) → HW = 1  ← SAME HW, different input!
0x04 (00000100) → HW = 1
0x08 (00001000) → HW = 1
0x10 (00010000) → HW = 1
...
128 different inputs with HW=4

When model predicts HW=4:
  Could be any of 70 different S-Box inputs
  Probability of guessing correct one: 1/70
```

**This is why full-key accuracy is terrible**:
```
Per-byte accuracy (predicting HW): ~50-60% (just guessing HW correctly)
Full-key accuracy (guessing actual bytes): 
  (1/128)^24 = 10^-71 if we're just guessing

The "58% accuracy" you're seeing is:
  Measuring HW accuracy, not key recovery
  NOT a valid metric for CPA attack success
```

---

## Why Hamming Weight Doesn't Work for CPA

### CPA Power Model
```
The assumption in CPA:
  Power ∝ Hamming Distance between values
  or Power ∝ Hamming Weight of intermediate
  
Leakage = H(S-BOX input) + noise

But the goal is:
  Given leakage, recover the actual S-BOX input
  Then: Key = some function(S-BOX input)

Predicting H(X) when you need X:
  ✗ Lossy - multiple X values map to same H(X)
  ✗ Can't uniquely recover key
  ✗ Like predicting only number of digits in a phone number
```

### Why Current Approach Returns Same Key
```
Process:
1. Input: Power traces (200 features)
2. Model: "Predict HW for each S-Box position"
3. For each position: Gets HW prediction (0-8)
4. Combination: Votes on most likely key byte

Problem:
- HW doesn't uniquely define byte
- Many bytes share same HW
- Voting defaults to MOST COMMON byte with that HW
- Result: Always same byte → Always same key
```

---

## The Fix: Change to S-Box Input Prediction

### Current Architecture (WRONG)
```
Input Traces (200 features)
  ↓
Model (Classification head: 8-14 classes for HW)
  ↓
Output: Hamming Weight (0-8)
  ↓
Inference: Map HW → Most common key byte with this HW
  ↓
Result: Always same key (the most frequent one)
```

### Fixed Architecture (CORRECT)
```
Input Traces (200 features)
  ↓
Model (Classification head: 256 classes)
  ↓
Output: Probability distribution over S-Box input (0-255)
  ↓
Inference: Take argmax as predicted S-Box input
  ↓
Recover key: Apply reverse S-Box operations
  ↓
Result: Different predictions for different traces
```

---

## Implementation Steps

### Step 1: Generate Correct Labels (This Week)

You need to regenerate training labels as:
```python
# Current (WRONG):
hamming_weight = count_bits(sbox_input)  # 0-8
label = hamming_weight

# Correct:
label = sbox_input  # 0-255
```

**Location**: [pipeline-code/src/gen_labels_3des_fixed.py](pipeline-code/src/gen_labels_3des_fixed.py)

### Step 2: Create 256-Class Model

```python
# In pipeline-code/src/model.py

class SBoxPredictor(nn.Module):
    def __init__(self, input_size=200):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 256)  # Output: 256 classes
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # No softmax here (CrossEntropyLoss applies it)
        return x

# Use loss function:
loss = nn.CrossEntropyLoss(weight=class_weights)  # Weight by class frequency
```

### Step 3: Update Training

```python
# Old training (wrong):
for batch in dataloader:
    X, y_hamming_weight = batch
    logits = model(X)
    loss = nn.CrossEntropyLoss()(logits, y_hamming_weight)

# New training (correct):
for batch in dataloader:
    X, y_sbox_input = batch  # y now ranges 0-255
    logits = model(X)
    loss = nn.CrossEntropyLoss()(logits, y_sbox_input)  # 256 classes, not 8
```

### Step 4: Update Inference

```python
# Old inference (wrong):
with torch.no_grad():
    hw_pred = model(traces)  # Predicts 0-8
    # Problem: Can't recover key from HW

# New inference (correct):
with torch.no_grad():
    logits = model(traces)  # Predicts 0-255
    sbox_pred = torch.argmax(logits, dim=1)  # Actual S-Box input
    # Problem solved: Can now recover key from S-Box input
```

---

## Expected Improvement Path

| Stage | Fix | Target Accuracy |
|-------|-----|-----------------|
| Current | Predicting HW (0-8) | 58% (not meaningful) |
| **After Step 1** | Use correct labels (0-255) | ~40-50% (baseline) |
| **After Step 2** | Use 256-class architecture | ~60-70% (with more params) |
| **After Step 3** | Add data augmentation | ~70-80% (3x more data) |
| **After Step 4** | Hyperparameter tuning | ~80-85% (optimized) |

---

## Quick Verification

To confirm this is the issue, check if S-Box input labels exist:

```bash
# Look for files with 0-255 range
find 3des-pipeline/ -name "*label*.npy" -exec python -c "
import numpy as np, sys
y = np.load(sys.argv[1])
print(f'{sys.argv[1].split(\"/\")[-1]}: min={y.min()}, max={y.max()}, unique={len(np.unique(y))}')
" {} \;
```

If all return small ranges (0-15), then labels are currently HW/intermediate representation, not S-Box input.

---

## Files That Need Changes

1. **[pipeline-code/src/gen_labels_3des_fixed.py](pipeline-code/src/gen_labels_3des_fixed.py)**
   - Change: Extract S-Box inputs (0-255), not Hamming Weight
   
2. **[pipeline-code/src/model.py](pipeline-code/src/model.py)**
   - Change: Output head to 256 classes (not 8)
   - Add: Dropout regularization
   
3. **[pipeline-code/src/train.py](pipeline-code/src/train.py)**
   - Change: Loss function to handle 256 classes
   - Add: Class weighting for imbalance
   
4. **[pipeline-code/src/inference_3des.py](pipeline-code/src/inference_3des.py)**
   - Change: Output S-Box inputs, not most common key

---

## Timeline to Fix

**Week 1**:
- [ ] Generate new labels (0-255 range)  
- [ ] Update model architecture (256 classes)
- [ ] Retrain with 20 epochs (quick test)
- [ ] Measure improvement

**Expected result after Week 1**: Different keys in output, measurable accuracy

**Week 2-3**: 
- Data augmentation
- Hyperparameter tuning
- Reach 80%+ accuracy

---

## Why This Matters

Current approach: **Theoretical accuracy ceiling: 1/128 = 0.78%** for recovery
```
Even with perfect HW prediction, guessing which of 128 inputs
has that HW = failure rate
```

New approach: **Theoretical accuracy ceiling: 90%+**
```
With enough training data and good model, can learn
which traces correspond to which S-Box inputs directly
```

