# CONFIDENCE SCORING FOR BLIND TRACES

## Why Confidence Matters for Greenvisa

```
Blind Trace Attack Scenario:
┌─ Input: Power trace from unknown card (Greenvisa)
├─ Unknown: Card type, key value, signature characteristics
├─ Process: Extract features → Ensemble prediction → Key recovery
└─ PROBLEM: How do we know if prediction is correct?

SOLUTION: Confidence score
└─ Tells us: "How certain are the models about this prediction?"
   ├─ 0.95 confidence → "95% sure this key is correct"
   ├─ 0.70 confidence → "70% sure (borderline, verify manually)"
   └─ 0.50 confidence → "Only 50-50 chance (discard/flag)"
```

---

## Implementation: Add Confidence to Your Inference Code

### Location: `pipeline-code/src/inference_3des.py`

Find the key recovery function and add confidence tracking:

```python
def attack_3des_ensemble(traces_features, models_3des, K1_recovered=None):
    """
    3DES ensemble attack with confidence scoring
    
    Args:
        traces_features: Pre-processed features for all traces
        models_3des: Dictionary of trained ensemble models
        K1_recovered: Recovered K1 (from stage 1), or None for blind
    
    Returns:
        recovered_keys: List of (key, confidence, debug_info) tuples
    """
    recovered_keys_all = []
    
    for trace_idx, trace_features in enumerate(traces_features):
        try:
            # Stage 1: Recover K1 (8 S-boxes × 3 key types)
            K1_sboxes, K1_confidence = recover_K1_with_confidence(
                trace_features, 
                models_3des['stage1']
            )
            
            # Stage 2: Recover K2 (using recovered K1)
            K2_sboxes, K2_confidence = recover_K2_with_confidence(
                trace_features,
                K1_sboxes,
                models_3des['stage2']
            )
            
            # Combine keys and compute final confidence
            full_key = combine_des_keys(K1_sboxes, K2_sboxes)
            final_confidence = compute_final_confidence(K1_confidence, K2_confidence)
            
            recovered_keys_all.append({
                'trace_idx': trace_idx,
                'KENC': full_key['KENC'],
                'KMAC': full_key['KMAC'],
                'KDEK': full_key['KDEK'],
                'confidence': final_confidence,
                'K1_confidence': K1_confidence,
                'K2_confidence': K2_confidence,
                'status': 'SUCCESS'
            })
            
        except Exception as e:
            recovered_keys_all.append({
                'trace_idx': trace_idx,
                'confidence': 0.0,
                'status': 'FAILED',
                'error': str(e)
            })
    
    return recovered_keys_all


def recover_K1_with_confidence(trace_features, models_stage1):
    """
    Recover K1 with confidence scores from ensemble
    
    Returns:
        K1_sboxes: Dict of recovered 6-bit keys for each S-box
        avg_confidence: Mean ensemble confidence (0.0-1.0)
    """
    
    K1_sboxes = {}
    confidence_per_sbox = {}
    
    # For each of 8 S-boxes
    for sbox_idx in range(8):
        sbox_predictions = {}
        vote_counts = {}
        
        # Get predictions from all ensemble members
        ensemble_size = len(models_stage1[sbox_idx])
        
        for member_idx, model in enumerate(models_stage1[sbox_idx]):
            # Predict: power_features → sbox_output_6bits
            sbox_output = model.predict(trace_features)[0]  # 0-63
            
            # Track vote
            if sbox_output not in vote_counts:
                vote_counts[sbox_output] = 0
            vote_counts[sbox_output] += 1
        
        # Winner: most voted sbox output
        winner_output = max(vote_counts.items(), key=lambda x: x[1])[0]
        K1_sboxes[f'sbox_{sbox_idx}'] = winner_output
        
        # Confidence: agreement score
        agreement = max(vote_counts.values()) / ensemble_size
        margin = (max(vote_counts.values()) - sorted(vote_counts.values(), reverse=True)[1]) / ensemble_size if len(vote_counts) > 1 else 1.0
        
        confidence = 0.7 * agreement + 0.3 * margin
        confidence_per_sbox[f'sbox_{sbox_idx}'] = confidence
    
    # Average confidence across all S-boxes
    avg_confidence = sum(confidence_per_sbox.values()) / len(confidence_per_sbox)
    
    return K1_sboxes, avg_confidence


def recover_K2_with_confidence(trace_features, K1_sboxes, models_stage2):
    """
    Recover K2 with confidence, using recovered K1
    Similar logic to recover_K1_with_confidence
    """
    
    K2_sboxes = {}
    confidence_per_sbox = {}
    
    for sbox_idx in range(8):
        vote_counts = {}
        ensemble_size = len(models_stage2[sbox_idx])
        
        for model in models_stage2[sbox_idx]:
            sbox_output = model.predict(trace_features)[0]
            if sbox_output not in vote_counts:
                vote_counts[sbox_output] = 0
            vote_counts[sbox_output] += 1
        
        winner_output = max(vote_counts.items(), key=lambda x: x[1])[0]
        K2_sboxes[f'sbox_{sbox_idx}'] = winner_output
        
        agreement = max(vote_counts.values()) / ensemble_size
        confidence = agreement
        confidence_per_sbox[f'sbox_{sbox_idx}'] = confidence
    
    avg_confidence = sum(confidence_per_sbox.values()) / len(confidence_per_sbox)
    
    return K2_sboxes, avg_confidence


def compute_final_confidence(K1_confidence, K2_confidence):
    """
    Final confidence is weighted average of Stage 1 and Stage 2 confidence
    
    Stage 1 is more important (if K1 is wrong, K2 will be wrong too)
    """
    return 0.6 * K1_confidence + 0.4 * K2_confidence
```

---

## Implementation: Output with Confidence

### Modify Output Generation

```python
def generate_output_with_confidence(recovered_keys, output_dir, threshold=0.70):
    """
    Generate output CSV with confidence filtering
    
    Args:
        recovered_keys: List of recovered key dicts with confidence
        output_dir: Where to save results
        threshold: Only output keys with confidence >= threshold
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate high vs low confidence
    high_confidence = [k for k in recovered_keys if k.get('confidence', 0) >= threshold]
    low_confidence = [k for k in recovered_keys if k.get('confidence', 0) < threshold]
    
    # Write high confidence (main output)
    with open(output_dir / 'Final_Report_high_confidence.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trace_idx', 'KENC', 'KMAC', 'KDEK', 'confidence', 'status'])
        writer.writeheader()
        for key in high_confidence:
            writer.writerow(key)
    
    # Write all results with confidence for analysis
    with open(output_dir / 'Final_Report_all_traces.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trace_idx', 'KENC', 'KMAC', 'KDEK', 'confidence', 'K1_confidence', 'K2_confidence', 'status'])
        writer.writeheader()
        for key in recovered_keys:
            writer.writerow(key)
    
    # Write low confidence for manual review
    if low_confidence:
        with open(output_dir / 'suspicious_low_confidence.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['trace_idx', 'KENC', 'KMAC', 'KDEK', 'confidence', 'status'])
            writer.writeheader()
            for key in low_confidence:
                writer.writerow(key)
        
        print(f"⚠ WARNING: {len(low_confidence)} traces have low confidence (<{threshold})")
        print(f"   See: {output_dir / 'suspicious_low_confidence.csv'}")
    
    print(f"✓ High confidence ({threshold}+): {len(high_confidence)} traces")
    print(f"✓ All results: {len(recovered_keys)} traces")
```

---

## Interpreting Confidence Scores

### For Blind Traces (Like Greenvisa)

```
CONFIDENCE SCORE INTERPRETATION FOR BLIND TRACES:

0.90-1.00 (HIGH CONFIDENCE):
├─ Meaning: Models are very certain about the prediction
├─ Action: Accept and report key
├─ Trust: 95%+ (can deploy directly)
└─ Risk: Very low

0.80-0.90 (MEDIUM CONFIDENCE):
├─ Meaning: Models are fairly confident
├─ Action: Accept but flag for monitoring
├─ Trust: 85-95%
└─ Risk: Low

0.70-0.80 (BORDERLINE CONFIDENCE):
├─ Meaning: Models have some uncertainty
├─ Action: Accept with reduced confidence
├─ Trust: 70-85%
└─ Risk: Medium - consider manual validation

< 0.70 (LOW CONFIDENCE):
├─ Meaning: Models are uncertain, likely incorrect
├─ Action: DO NOT trust, require additional validation
├─ Trust: <70%
└─ Risk: High - may be completely wrong
   Possible causes:
   ├─ Card type not seen during training (card-specific pattern)
   ├─ Unusual power leakage signature
   ├─ Noisy/corrupted trace
   └─ Models overfit to training data
```

### Practical Thresholds by Use Case

```
USE CASE                STRICT   RELAXED   ACTION FOR LOW
─────────────────────────────────────────────────────────────
Development/Testing     0.70     0.50      Log for analysis
Quality Assurance       0.80     0.70      Flag for review
Production Attack       0.85     0.75      Return "UNCERTAIN"
Critical Applications   0.90     0.80      Require 2nd attack
```

---

## What Causes Low Confidence in Blind Traces?

### Scenario 1: Card Not Seen During Training
```
Training: Mastercard + Visa keys
Test: Greenvisa (not in training)

Expected behavior:
- If models generalize: Confidence 0.85+ (learned S-box patterns)
- If models overfit: Confidence 0.50 (unfamiliar leakage signature)

Diagnostic:
├─ Confidence drops sharply for Greenvisa
├─ But works well for Visa/Mastercard
└─ Conclusion: Models are card-type specific
   Solution: Retrain with Greenvisa samples
```

### Scenario 2: Unusual Leakage Profile
```
Traces: Different hardware generation, different EMI filtering, etc.

Expected:
├─ Some traces have high confidence (normal leakage)
├─ Some have low confidence (unusual pattern)
└─ Average confidence < 0.70

Diagnosis:
├─ Check trace quality
├─ Verify preprocessing (POI extraction still valid?)
└─ May need special handling for this batch
```

### Scenario 3: Trace Noise/Corruption
```
Single trace: Extremely noisy, partially corrupted

Expected:
├─ That one trace: confidence 0.40
├─ Neighboring traces: confidence 0.85+
└─ Pattern: Sporadic low-confidence spikes

Action:
├─ Filter out traces with confidence < 0.70
├─ Require reacquisition of low-confidence traces
└─ Trust rest of dataset
```

---

## Confidence-Based Post-Processing

### For Greenvisa Attack Results

```python
def post_process_results_with_confidence(results_csv, strict_threshold=0.85, output_dir='processed'):
    """
    Process attack results with confidence filtering
    """
    
    import pandas as pd
    
    df = pd.read_csv(results_csv)
    
    # Filter by confidence
    high_conf = df[df['confidence'] >= strict_threshold]
    medium_conf = df[(df['confidence'] >= 0.70) & (df['confidence'] < strict_threshold)]
    low_conf = df[df['confidence'] < 0.70]
    
    # Statistics
    print(f"""
CONFIDENCE BREAKDOWN:
├─ High (≥{strict_threshold}): {len(high_conf)} traces ({len(high_conf)/len(df):.1%})
├─ Medium (0.70-{strict_threshold}): {len(medium_conf)} traces ({len(medium_conf)/len(df):.1%})
└─ Low (<0.70): {len(low_conf)} traces ({len(low_conf)/len(df):.1%})

RECOMMENDATION:
├─ Deploy high confidence keys: {len(high_conf)} traces
├─ Review medium confidence: {len(medium_conf)} traces (manual validation)
└─ Discard/reinject low confidence: {len(low_conf)} traces
""")
    
    # Export by category
    high_conf.to_csv(f'{output_dir}/high_confidence_keys.csv', index=False)
    medium_conf.to_csv(f'{output_dir}/medium_confidence_keys.csv', index=False)
    low_conf.to_csv(f'{output_dir}/low_confidence_keys.csv', index=False)
    
    return high_conf, medium_conf, low_conf
```

---

## Validation: Confidence vs Accuracy

### How to Verify Confidence Scores Are Meaningful

During Stage 2 (Visa test) or Stage 3 (Greenvisa test):

```python
def validate_confidence_calibration(predictions_with_conf, ground_truth_keys):
    """
    Verify that confidence scores actually predict accuracy
    
    Good calibration: High confidence → High accuracy
    Bad calibration: Confidence uncorrelated with accuracy
    """
    
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    # Compute accuracy for each prediction
    accuracies = []
    confidences = []
    
    for pred in predictions_with_conf:
        actual_key = ground_truth_keys[pred['trace_idx']]
        is_correct = (pred['KENC'] == actual_key['KENC'])
        
        accuracies.append(1.0 if is_correct else 0.0)
        confidences.append(pred['confidence'])
    
    # Verify correlation
    correlation = np.corrcoef(confidences, accuracies)[0, 1]
    auc = roc_auc_score(accuracies, confidences)
    
    print(f"""
CONFIDENCE CALIBRATION CHECK:
├─ Pearson correlation (conf vs accuracy): {correlation:.3f}
│  └─ Should be > 0.7 for good calibration
├─ ROC-AUC: {auc:.3f}
│  └─ Should be > 0.8 for good calibration
└─ Status: {"✓ GOOD" if correlation > 0.7 else "✗ BAD (confidence scores unreliable)"}
""")
    
    # Break down by confidence bins
    bins = [(0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    print("\nACCURACY BY CONFIDENCE BIN:")
    for low, high in bins:
        mask = (np.array(confidences) >= low) & (np.array(confidences) < high)
        bin_accuracy = np.mean(np.array(accuracies)[mask]) if mask.any() else 0
        bin_count = np.sum(mask)
        print(f"  {low:.2f}-{high:.2f}: {bin_accuracy:.1%} accuracy ({bin_count} traces)")
```

---

## Summary: Confidence for Blind Traces

| Threshold | Coverage | Trust Level | Deployment |
|-----------|----------|------------|-----------|
| **0.90** | ~70% | Very High (95%+) | Direct deploy |
| **0.85** | ~80% | High (90%+) | Production |
| **0.75** | ~90% | Medium (80%+) | With monitoring |
| **0.70** | ~95% | Lower (70%+) | All traces, flag low |
| **No threshold** | 100% | Variable | Research only |

### For Greenvisa Specifically:
- Start with **strict 0.85** to identify high-confidence predictions
- If > 80% coverage with > 95% accuracy → **Excellent generalization**
- Then use **relaxed 0.70** for full coverage
- Any **confidence < 0.70** should trigger investigation

---

## Next Steps

1. **Implement confidence scoring** in your inference code (copy code above)
2. **Run Stage 1** with confidence output enabled
3. **Run Stage 2** and verify confidence correlates with accuracy
4. **Run Stage 3 Greenvisa** with both 0.85 and 0.70 thresholds
5. **Analyze results**: Does confidence predict accuracy?
6. **Deploy**: If confidence is well-calibrated, use threshold-based filtering

This ensures blind traces (like Greenvisa) can be attacked with **confidence-based risk assessment** rather than blind faith in model predictions.
