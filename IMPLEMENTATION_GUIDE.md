# Implementation Guide: Fix the 3DES ML Pipeline

## Executive Summary
- **Problem**: Model predicts Hamming Weight, not S-Box inputs
- **Impact**: Can't recover actual keys (58% is not meaningful)
- **Fix Timeline**: 2-3 weeks to reach 85%+ accuracy
- **Effort**: Medium (2-3 python scripts to modify)

---

## Phase 1: Verify the Problem (TODAY)

### Step 1.1: Check Current Labels
```bash
cd i:\freelance\SCA\ Smartcard\ ML\ Pipeline-3des
python -c "
import numpy as np
from pathlib import Path

# Check all label files
for f in Path('3des-pipeline/Processed/3des').glob('Y_*.npy'):
    y = np.load(f)
    unique = len(np.unique(y))
    if unique < 50:  # Likely HW or intermediate
        print(f'{f.name}: {unique} unique values (range: {y.min()}-{y.max()})')
        print(f'  -> This looks like Hamming Weight, NOT S-Box input')
        print(f'  -> PROBLEM CONFIRMED')
    elif unique > 200:  # Likely S-Box input
        print(f'{f.name}: {unique} unique values (range: {y.min()}-{y.max()})')
        print(f'  -> Looks like S-Box input (0-255)')
"
```

**Expected Result**: All files show <50 unique values → Problem confirmed

### Step 1.2: Check Original Data
```bash
# Look for raw CSV files that might have original labels
ls -R 3des-pipeline/Input/

# If CSVs exist, check what label was extracted:
python -c "
import pandas as pd
df = pd.read_csv('3des-pipeline/Input/mastercard_0.csv', nrows=100)
# Check if there's indication of what was predicted
# Look for columns with 'key' or 'label'
print(df.head())
"
```

---

## Phase 2: Generate Correct Labels (WEEK 1)

### Step 2.1: Review Current Label Generation

Open this file:
**[pipeline-code/src/gen_labels_3des_fixed.py](pipeline-code/src/gen_labels_3des_fixed.py)**

Current code probably looks like:
```python
# WRONG:
sbox_output = des_sbox(key_byte, plaintext_byte)
hamming_weight = bin(sbox_output).count('1')  # Counts bits
labels.append(hamming_weight)  # Range: 0-8

# CORRECT:
sbox_input = plaintext_byte XOR key_byte  # Or similar
labels.append(sbox_input)  # Range: 0-255
```

### Step 2.2: Find Where S-Box Input is Calculated

Search for the step where interference is prepared:
```bash
cd pipeline-code/src
grep -n "sbox\|hamming\|leakage" gen_labels_3des_fixed.py
```

Look for:
- Where `plaintext XOR guessed_key` is computed
- Where S-Box is applied
- Where output is converted to label

### Step 2.3: Modify to Extract S-Box Input

The fix should be simple - instead of extracting a property (HW), extract the value:

```python
# File: pipeline-code/src/gen_labels_3des_fixed.py

def extract_sbox_input(round_num, byte_pos, key, ciphertext):
    """
    Extract the S-Box input for a specific position
    
    3DES attack target:
      1. Decrypt with known key guess
      2. Look at S-Box inputs to first DES round
      3. S-Box input = plaintext[byte_pos] XOR subkey[byte_pos]
      
    Returns: S-Box input value (0-255)
    """
    # Get first round subkey
    subkey = get_round_subkey(key, round_num, byte_pos)
    
    # Get plaintext for this position (from decryption)
    plaintext_byte = decrypt_get_intermediate(ciphertext, key, pos=byte_pos)
    
    # S-Box input = XOR
    sbox_input = plaintext_byte ^ subkey
    
    return sbox_input  # Range: 0-255

# Then in main processing loop:
for trace_idx, trace in enumerate(traces):
    labels_for_this_trace = []
    for byte_pos in range(24):  # 3DES = 24 bytes
        sbox_in = extract_sbox_input(round=1, byte_pos=byte_pos, key=key, ciphertext=ciphertext)
        labels_for_this_trace.append(sbox_in)
    all_labels.append(labels_for_this_trace)

# Save as new file
np.save('3des-pipeline/Processed/3des/Y_SBOX_INPUTS.npy', np.array(all_labels))
```

### Step 2.4: Generate New Dataset

```bash
# After modifying gen_labels_3des_fixed.py:
cd pipeline-code
python src/gen_labels_3des_fixed.py --output-sbox-inputs

# Verify new labels
python -c "
import numpy as np
y = np.load('3des-pipeline/Processed/3des/Y_SBOX_INPUTS.npy')
print(f'New label shape: {y.shape}')
print(f'Expected: (n_traces, 24) for 3DES bytes')
print(f'Range per byte: 0-255')
print(f'Actual: {y.min()}-{y.max()}')
print(f'Unique values per byte (first 3): {[len(np.unique(y[:, i])) for i in range(3)]}')
"
```

**Expected Output**:
```
New label shape: (10000, 24)
Unique values per byte (first 3): [256, 256, 256]
```

---

## Phase 3: Update Model Architecture (WEEK 1)

### Step 3.1: Review Current Model

Open: **[pipeline-code/src/model.py](pipeline-code/src/model.py)**

Look for the output layer. Current probably looks like:
```python
# WRONG:
self.output = nn.Linear(hidden_dim, 8)  # or 14, or 16 (for HW range)

# CORRECT:
self.output = nn.Linear(hidden_dim, 256)  # 256 classes for S-Box input
```

### Step 3.2: Update Architecture

Make these changes:

```python
# File: pipeline-code/src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SBoxInputPredictor(nn.Module):
    """
    Predicts the S-Box input (0-255) given power trace features
    
    Architecture:
    - Input: 200 features (preprocessed power trace)
    - Hidden: 256 neurons + Dropout
    - Hidden: 256 neurons + Dropout
    - Output: 256 classes (S-Box input values)
    """
    
    def __init__(self, input_size=200, hidden_size=256):
        super().__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        # Hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output layer: 256 classes for S-Box input
        self.fc3 = nn.Linear(hidden_size, 256)
        
    def forward(self, x):
        # First hidden layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output: logits (no softmax - CrossEntropyLoss applies it)
        x = self.fc3(x)
        
        return x

# Usage:
model = SBoxInputPredictor(input_size=200, hidden_size=256)
```

### Step 3.3: Add Class Weighting

S-Box inputs are likely imbalanced in training data. Add weighting:

```python
# In training file (e.g., pipeline-code/src/train.py)

import torch
import numpy as np
from torch.utils.data import DataLoader

def compute_class_weights(labels):
    """
    Compute weights for imbalanced classes
    
    If some S-Box inputs appear more often, weight rare ones higher
    """
    # Count occurrences of each class (0-255)
    unique, counts = np.unique(labels, return_counts=True)
    
    # Weight = 1 / frequency
    weights = np.zeros(256)
    weights[unique] = 1.0 / counts
    
    # Normalize so average weight = 1
    weights = weights / weights.mean()
    
    return torch.from_numpy(weights).float()

# During training:
labels = np.load('path/to/Y_SBOX_INPUTS.npy')
class_weights = compute_class_weights(labels.flatten())

# Loss function with weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## Phase 4: Retrain Model (WEEK 1)

### Step 4.1: Update Training Script

Modify **[pipeline-code/src/train.py](pipeline-code/src/train.py)**:

```python
# Key changes:

# 1. Load correct labels
y_labels = np.load('3des-pipeline/Processed/3des/Y_SBOX_INPUTS.npy')  # NEW

# 2. Use 256-class loss
from torch.optim.lr_scheduler import CosineAnnealingLR

model = SBoxInputPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 3. Training loop
best_val_acc = 0
patience = 15

for epoch in range(max_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        logits = model(batch_X)
        
        # For each byte position
        loss = 0
        for byte_idx in range(24):
            loss += criterion(logits, batch_y[:, byte_idx])
        loss = loss / 24  # Average over bytes
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            logits = model(batch_X)
            preds = torch.argmax(logits, dim=1)
            val_acc += (preds == batch_y[:, 0]).sum().item()  # First byte
    
    val_acc /= len(val_loader.dataset)
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping")
        break
    
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
```

### Step 4.2: Run Training

```bash
cd pipeline-code

# Quick test (small dataset)
python src/train.py --epochs=20 --batch-size=32 --test-mode

# Full training
python src/train.py --epochs=100 --batch-size=32
```

**Expected Timeline**:
- First 5 epochs: Loss decreases, accuracy improves
- Epoch 20: ~60-70% accuracy (for first byte)
- Epoch 50: ~75-80% accuracy
- Epoch 100: ~80-85% accuracy

---

## Phase 5: Verify & Measure Improvement (WEEK 1)

### Step 5.1: Test on Known Dataset

```bash
# Create test script
python -c "
import numpy as np
import torch

# Load test data
X_test = np.load('3des-pipeline/Processed/3des/X_features.npy')
y_test = np.load('3des-pipeline/Processed/3des/Y_SBOX_INPUTS.npy')

# Load trained model
model = SBoxInputPredictor()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Predict
X_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    logits = model(X_tensor)
    predictions = torch.argmax(logits, dim=1)

# Accuracy per byte
accuracies = []
for byte_idx in range(24):
    pred_byte = predictions[:, byte_idx] if predictions.dim() > 1 else predictions
    actual_byte = y_test[:, byte_idx]
    acc = (pred_byte == actual_byte).mean()
    accuracies.append(acc)
    print(f'Byte {byte_idx}: {acc:.1%}')

print(f'Average: {np.mean(accuracies):.1%}')
print(f'Mean ± Std: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}')
"
```

**Expected Result**:
- Average accuracy: 70-80% per byte
- Different predictions for different traces
- Variation across byte positions

### Step 5.2: Generate Final Report

```bash
# Generate predictions on test set
python pipeline-code/src/inference_3des.py \
  --model best_model.pth \
  --output 3des-pipeline/Output/Report_After_Fix.csv

# Compare with baseline
python -c "
import pandas as pd
baseline = pd.read_csv('3des-pipeline/Output/Final_Report_mastercard_session.csv')
after_fix = pd.read_csv('3des-pipeline/Output/Report_After_Fix.csv')

print('Baseline (wrong labels):')
print(f'  Unique keys: {baseline[\"3DES_KENC\"].nunique()}')

print('After fix (correct labels):')
print(f'  Unique keys: {after_fix[\"3DES_KENC\"].nunique()}')
print(f'  Improvement: {after_fix[\"3DES_KENC\"].nunique()} different predictions')
"
```

---

## Phase 6: Data Augmentation (WEEK 2)

Once labels and model are working, augment data:

```python
# pipeline-code/src/data_augmentation.py

def augment_traces(traces, multiplier=3):
   \"\"\"Generate augmented traces\"\"\"
    augmented = [traces]
    
    for _ in range(multiplier - 1):
        # Gaussian noise
        noise = np.random.normal(0, 0.01, traces.shape)
        augmented.append(traces + noise)
        
        # Time shift
        shift = np.random.randint(-3, 4)
        augmented.append(np.roll(traces, shift, axis=1))
        
        # Amplitude scaling
        scale = np.random.uniform(0.95, 1.05, (traces.shape[0], 1))
        augmented.append(traces * scale)
    
    return np.vstack(augmented)

# Usage:
X_aug = augment_traces(X_train, multiplier=3)
y_aug = np.repeat(y_train, 3, axis=0)

# Retrain with 3x more data
# Expected: +5-10% accuracy
```

---

## Checklist

### Week 1
- [ ] Verify problem (check label ranges)
- [ ] Modify label generation script
- [ ] Generate new Y_SBOX_INPUTS.npy file
- [ ] Update model architecture (256 classes)
- [ ] Add class weighting
- [ ] Retrain (20-50 epochs)
- [ ] Measure improvement (target: 70-80% per byte)
- [ ] Verify different predictions in output

### Week 2
- [ ] Implement data augmentation (3x multiplier)
- [ ] Retrain with augmented data
- [ ] Measure improvement (target: 75-85%)
- [ ] Hyperparameter tuning

### Week 3
- [ ] Cross-validation (5-fold)
- [ ] Final test set evaluation
- [ ] Generate accuracy report
- [ ] Document results

---

## Troubleshooring

### Issue: Accuracy still low after fix
**Cause**: Data quality or feature extraction problem
**Solution**:
1. Check if power traces are properly preprocessed
2. Verify S-Box input extraction is correct
3. Check for class imbalance (>10x difference in counts)

### Issue: Model training diverges
**Cause**: Learning rate too high
**Solution**:
1. Reduce learning rate: 1e-3 → 1e-4
2. Use learning rate scheduler
3. Add more dropout (0.3 → 0.5)

### Issue: Overfitting (train acc >> val acc)
**Cause**: Model too large for dataset size
**Solution**:
1. Reduce hidden size: 256 → 128
2. Increase dropout: 0.3 → 0.5
3. Add L2 regularization: weight_decay=1e-4

---

## Questions?

If stuck, check these files:
- Y_SBOX_INPUTS.npy verification
- Model forward pass (256 output dimension)
- CrossEntropyLoss handling 256 classes
- Inference code outputting S-Box input predictions

