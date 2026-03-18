# STAGED VALIDATION & DEPLOYMENT PLAN

## Overview
After retraining with 100% key coverage, you'll validate models through a **progressive testing strategy** before attacking Greenvisa traces (blind, unknown structure).

```
STAGE 1: Mastercard (3DES)     [Known cards, known keys]
   ↓ (if 100% accuracy)
STAGE 2: Visa + Mastercard     [Multiple card types, validate diversity]
   (RSA testing)               
   ↓ (if > 95% accuracy both)
STAGE 3: Greenvisa Blind Attack [Unknown card, confidence-based]
```

---

## STAGE 1: Masters Card 3DES Validation

### Objective
Verify retrained 3DES models work correctly on the card type they were trained on.

### Test Data
- **Source:** Mastercard traces from KALKI TEST CARD.xlsx
- **Key:** 9E15204313F... (known, with labels)
- **Type:** Labeled traces (for validation purposes)

### Validation Commands

```bash
# Run attack on Mastercard 3DES test set
python pipeline-code/main.py \
  --mode attack \
  --scan_type 3des \
  --card_type mastercard \
  --input_dir "I:\freelance\SCA-Smartcard-Pipeline-3\Input1" \
  --models_dir "3des-pipeline/models/3des" \
  --output_dir "3des-pipeline/Output/validation_stage_1_mastercard" \
  --confidence_threshold 0.8

# Expected output: validation_stage_1_mastercard/Final_Report_mastercard_session.csv
```

### Acceptance Criteria ✓
- **Key Recovery Accuracy:** ≥ 99% (models were trained on this card)
- **Confidence Scores:** Average > 0.85
- **Consistency:** Same trace produces same key every run
- **Stage 1 PASS:** All criteria met → Proceed to Stage 2

### Troubleshooting if FAIL ✗
```
If accuracy < 95%:
  → Models didn't retrain properly
  → Check: ingest.py uses external labels correctly
  → Re-run retrain_with_full_coverage.py
  → Verify KALKI file hasn't changed

If confidence < 0.70:
  → Models have high uncertainty
  → Increase epochs in retrain (--epochs 300)
  → Use more diverse training data (expand KALKI)
```

### Success Metrics
```
✓ MASTERCARD 3DES STAGE 1 PASS:
  Accuracy: 99.2%
  Avg Confidence: 0.87
  Failed keys: 0/500
  → Ready for Stage 2
```

---

## STAGE 2: RSA Multi-Card Validation

### Objective
Verify RSA models work across **both Visa AND Mastercard** to prove generalization.

### Test Data
- **Mastercard RSA:** 2315... key from KALKI TEST CARD.xlsx (known traces, labeled)
- **Visa RSA:** 23152... key from KALKI TEST CARD.xlsx (known traces, labeled)
- **Purpose:** Verify models learned RSA patterns, not card-specific signatures

### Validation Commands

```bash
# Test 1: Generate Mastercard RSA baseline
python pipeline-code/main.py \
  --mode attack \
  --scan_type rsa \
  --card_type mastercard \
  --input_dir "I:\freelance\SCA-Smartcard-Pipeline-3\Input1" \
  --models_dir "3des-pipeline/models/rsa" \
  --output_dir "3des-pipeline/Output/validation_stage_2a_rsa_mastercard" \
  --confidence_threshold 0.75

# Test 2: Cross-card validation - Mastercard models on Visa traces
python pipeline-code/main.py \
  --mode attack \
  --scan_type rsa \
  --card_type visa \
  --input_dir "I:\freelance\SCA-Smartcard-Pipeline-3\Input1" \
  --models_dir "3des-pipeline/models/rsa" \
  --output_dir "3des-pipeline/Output/validation_stage_2b_rsa_visa_blind" \
  --confidence_threshold 0.75 \
  --cross_card_test true  # Models trained on Mastercard, testing on Visa

# Test 3: Visa models on Mastercard (if trained on both)
python pipeline-code/main.py \
  --mode attack \
  --scan_type rsa \
  --card_type mastercard \
  --input_dir "I:\freelance\SCA-Smartcard-Pipeline-3\Input1" \
  --models_dir "3des-pipeline/models/rsa" \
  --output_dir "3des-pipeline/Output/validation_stage_2c_rsa_mastercard_blind" \
  --confidence_threshold 0.75 \
  --cross_card_test true  # Models trained on Visa, testing on Mastercard
```

### Acceptance Criteria ✓

**Mastercard RSA (Baseline):**
- Accuracy: ≥ 98%
- Avg Confidence: ≥ 0.80
- Status: Golden baseline

**Visa RSA (Cross-Card Test):**
- Accuracy: ≥ 95% (allowing 3% cross-card degradation)
- Avg Confidence: ≥ 0.75
- Key Components Recovery:
  - p: ≥ 95%
  - q: ≥ 95%
  - dp: ≥ 90%
  - dq: ≥ 90%
  - qinv: ≥ 85%

**Stage 2 PASS:** Both Mastercard ≥ 95% AND Visa ≥ 93% → Proceed to Stage 3

### Interpretation

```
GOOD GENERALIZATION SIGNS:
├─ Mastercard: 98%, Visa: 96%  → ✓ Excellent (slight card-type variation)
├─ Mastercard: 98%, Visa: 94%  → ✓ Good (expected variation)
└─ Mastercard: 98%, Visa: 91%  → ✓ Acceptable (still generalizing)

BAD GENERALIZATION SIGNS:
├─ Mastercard: 98%, Visa: 80%  → ✗ Models overfit to Mastercard
├─ Mastercard: 98%, Visa: 50%  → ✗ Complete failure on Visa (memorization)
└─ High variance in key components → ✗ Some S-boxes not generalizing
```

### Troubleshooting if FAIL ✗

```
If Visa accuracy < 90%:
  → Models trained on mixed Visa/Mastercard but test shows bias
  → Reason: Probably used Mastercard-heavy training
  → Solution: Retrain with balanced Visa/Mastercard split
    python retrain_with_full_coverage.py --epochs 300 --balance_cards true

If Visa confidence < 0.70:
  → Models uncertain on cross-card traces
  → Reason: Not enough diversity in training
  → Solution: Expand KALKI file with 5-10 keys per card type

If specific key components fail (e.g., qinv):
  → That S-box doesn't generalize well
  → Solution: Increase model capacity for that component
           or use transfer learning from working components
```

---

## STAGE 3: Greenvisa Blind Attack

### Objective
Attack **completely unknown card type** (Greenvisa) using trained models + confidence scoring.
This is the **true generalization test** - if Stage 2 passes but Stage 3 fails, something is wrong.

### Test Data
- **Source:** Greenvisa traces from your input directory
- **Key:** UNKNOWN (this is a blind attack)
- **Challenge:** Models never saw Greenvisa during training
- **Validation:** Can we match recovered keys against known Greenvisa keys?

### Validation Commands

```bash
# Stage 3A: Preview - run attack with LOW confidence threshold
# This shows which keys the model is CERTAIN about
python pipeline-code/main.py \
  --mode attack \
  --scan_type 3des \
  --card_type greenvisa \
  --input_dir "I:\freelance\SCA-Smartcard-Pipeline-3\Input1" \
  --models_dir "3des-pipeline/models/3des" \
  --output_dir "3des-pipeline/Output/validation_stage_3a_greenvisa_strict" \
  --confidence_threshold 0.85 \
  --output_confidence true  # Include confidence scores in output

# Stage 3B: Full attack - include lower-confidence predictions
python pipeline-code/main.py \
  --mode attack \
  --scan_type 3des \
  --card_type greenvisa \
  --input_dir "I:\freelance\SCA-Smartcard-Pipeline-3\Input1" \
  --models_dir "3des-pipeline/models/3des" \
  --output_dir "3des-pipeline/Output/validation_stage_3b_greenvisa_full" \
  --confidence_threshold 0.70 \
  --output_confidence true
```

### Acceptance Criteria ✓

```
STRICT CONFIDENCE (threshold 0.85):
├─ Coverage: > 80% of traces processed
├─ Accuracy: ≥ 95% (on traces that model is confident about)
├─ Keys recovered: All KENC/KMAC/KDEK components
└─ Status: HIGH CONFIDENCE predictions only

FULL COVERAGE (threshold 0.70):
├─ Coverage: > 95% of traces processed
├─ Accuracy: ≥ 90% (on all traces, even with lower confidence)
├─ Failed predictions: < 5% completely wrong
└─ Status: All predictions attempted, confidence-scored

STAGE 3 PASS CRITERIA:
✓ Strict 0.85: Accuracy ≥ 95% AND Coverage > 80%
✓ Full 0.70: Accuracy ≥ 90% AND Coverage > 95%
✓ No systematic biases (keys consistently recoverable)
✓ Confidence scores correlate with accuracy
  (high confidence predictions have higher accuracy)
```

### Interpretation

```
EXCELLENT GENERALIZATION:
├─ Mastercard 3DES: 99%
├─ Visa RSA: 95%
├─ Greenvisa 3DES strict (0.85): 96% on 85% of traces
├─ Greenvisa 3DES full (0.70): 92% on 98% of traces
└─ RECOMMENDATION: ✓ SAFE FOR PRODUCTION

GOOD GENERALIZATION:
├─ Mastercard 3DES: 98%
├─ Visa RSA: 93%
├─ Greenvisa 3DES strict (0.85): 93% on 70% of traces
├─ Greenvisa 3DES full (0.70): 88% on 95% of traces
└─ RECOMMENDATION: ✓ OK FOR PRODUCTION (with monitoring)

POOR GENERALIZATION:
├─ Mastercard 3DES: 99%
├─ Visa RSA: 89%  ← ✗ Too much degradation
├─ Greenvisa 3DES strict (0.85): 85% on 40% of traces  ← ✗ Low coverage
├─ Greenvisa 3DES full (0.70): 75% on 90% of traces   ← ✗ Low accuracy
└─ RECOMMENDATION: ✗ NOT READY (need more training)

CATASTROPHIC FAILURE:
├─ Mastercard 3DES: 99%
├─ Visa RSA: 50%  ← ✗ Complete failure
├─ Greenvisa 3DES full (0.70): 40% on 95% of traces  ← ✗ Mostly wrong
└─ RECOMMENDATION: ✗ CRITICAL ISSUE (models memorizing, not generalizing)
                      Review architecture and retrain with more diverse data
```

### What If Stage 3 Fails?

```
Scenario 1: Greenvisa accuracy < 80% (poor generalization)
├─ Cause: Models overfit to Mastercard/Visa patterns
├─ Fix: Expand KALKI with Greenvisa samples (5-10 keys)
├─ Then: Retrain with --balance_cards true
└─ Revalidate: Run Stage 3 again

Scenario 2: Greenvisa accuracy 85-90% (acceptable)
├─ Status: Reasonable generalization
├─ Action: Deploy with confidence threshold 0.80
├─ Monitor: Track accuracy on real Greenvisa attacks
└─ Note: May need future retraining with more Greenvisa samples

Scenario 3: Greenvisa accuracy 91-95% (good)
├─ Status: Good generalization achieved
├─ Action: Deploy normally
├─ Monitor: Standard monitoring procedures
└─ Note: Production ready

Scenario 4: Greenvisa accuracy < 70% (critical failure)
├─ Cause: Possible data distribution shift
├─ Cause: Greenvisa has different leakage profile
├─ Action: STOP - do not deploy
├─ Investigation: 
│  - Check Greenvisa trace quality
│  - Verify key extraction is correct
│  - Analyze feature distributions
└─ Option: Collect Greenvisa training samples first
```

---

## CONFIDENCE SCORING IMPLEMENTATION

### Why Confidence Matters for Blind Traces

```
Blind Trace (Greenvisa):
├─ Input: Power trace, unknown card type, unknown key
├─ Models predict: S-box outputs
├─ Reconstruct: Key that produces those outputs
├─ PROBLEM: What if models are wrong?
│
└─ SOLUTION: Confidence score tells us model's uncertainty
   ├─ If confidence 0.95 → "Model is 95% sure"
   ├─ If confidence 0.60 → "Model is only 60% sure"
   └─ Flag low-confidence predictions for manual review
```

### Recommended Thresholds

```
APPLICATION      THRESHOLD   ACTION IF BELOW
─────────────────────────────────────────────────────
Development      0.70        Log for analysis
Staging/QA       0.80        Flag for manual review
Production       0.85        Return UNCERTAIN status
Critical Attack  0.90        Require additional validation

GREENVISA SPECIFIC:
├─ Stage 3A (Strict):  0.85 → Only process certain predictions
├─ Stage 3B (Full):    0.70 → Process all, but track confidence
└─ Monitor: Watch for accuracy vs confidence correlation
```

### Confidence Score Computation

```python
# In your inference_3des.py / inference_rsa.py:

def compute_confidence_score(ensemble_predictions):
    """
    Confidence = agreement + consistency
    
    High confidence:
    - All ensemble members predict same key → agreement = 1.0
    - Same trace produces same key every run → consistency = 1.0
    
    Low confidence:
    - Ensemble members disagree → agreement = 0.5
    - Same trace produces different keys → consistency = 0.6
    
    Final score = 0.5 * agreement + 0.5 * consistency
    """
    
    # 1. Ensemble agreement
    predicted_keys = [m.predict(trace) for m in ensemble]
    votes = Counter(predicted_keys)
    agreement = max(votes.values()) / len(ensemble)
    
    # 2. Prediction margin (how confident was the top vote)
    vote_margin = (max(votes.values()) - sorted(votes.values(), reverse=True)[1]) / len(ensemble)
    
    # Final confidence
    confidence = 0.7 * agreement + 0.3 * vote_margin
    return confidence, agreement, vote_margin
```

---

## COMPLETE VALIDATION CHECKLIST

### Before Starting Validation
- [ ] Retraining completed with 100% key coverage
- [ ] Models saved to `3des-pipeline/models/3des/` and `models/rsa/`
- [ ] All input traces available in `I:\freelance\SCA-Smartcard-Pipeline-3\Input1`
- [ ] KALKI TEST CARD.xlsx confirmed to have correct keys
- [ ] Greenvisa traces identified and ready
- [ ] Output directories created for each stage

### Stage 1: Mastercard 3DES
- [ ] Run Mastercard attack command
- [ ] Check Final_Report_mastercard_session.csv
- [ ] Accuracy calculation: matched_keys / total_keys
- [ ] Record in `validation_results.csv`:
  ```
  Stage,CardType,Scan,Accuracy,AvgConfidence,Status
  1,Mastercard,3DES,99.2,0.87,PASS
  ```
- [ ] If accuracy ≥ 99%: **PROCEED TO STAGE 2**
- [ ] If accuracy < 95%: **STOP, DEBUG, RETRAIN**

### Stage 2: RSA Multi-Card  
- [ ] Run Mastercard RSA baseline
- [ ] Run Visa RSA cross-card test
- [ ] Compare accuracies (note degradation)
- [ ] Check per-component accuracy (p, q, dp, dq, qinv)
- [ ] Record results:
  ```
  Stage,CardType,Scan,Accuracy,ComponentAccuracy,Status
  2,Mastercard,RSA,98.1,p:98%|q:97%|dp:96%|dq:95%|qinv:93%,PASS
  2,Visa,RSA,95.2,p:96%|q:95%|dp:93%|dq:92%|qinv:89%,PASS
  ```
- [ ] If Visa ≥ 93%: **PROCEED TO STAGE 3**
- [ ] If Visa < 90%: **EXPAND KALKI, RETRAIN, RETEST**

### Stage 3: Greenvisa Blind Attack
- [ ] Run strict confidence (0.85) attack
- [ ] Run full coverage (0.70) attack
- [ ] Measure coverage: processed_traces / total_traces
- [ ] Measure accuracy: recovered_keys_match_known / processed_traces
- [ ] Analyze confidence correlation:
  ```
  For confident predictions (≥ 0.85): accuracy should be ≥ 95%
  For all predictions (≥ 0.70): accuracy should be ≥ 90%
  ```
- [ ] Record results:
  ```
  Stage,CardType,Scan,Confidence,Coverage,Accuracy,Status
  3,Greenvisa,3DES,0.85,82%,96.1%,PASS(STRICT)
  3,Greenvisa,3DES,0.70,98%,91.2%,PASS(FULL)
  ```
- [ ] If passes criteria: **PRODUCTION READY**
- [ ] If fails: **EXPAND TRAINING DATA, RETRAIN, RETEST**

### Final Sign-Off
- [ ] All 3 stages passed with required accuracy
- [ ] Confidence scores properly computed and tracked
- [ ] No systematic biases in attacked keys
- [ ] Documentation complete with results
- [ ] **APPROVED FOR PRODUCTION DEPLOYMENT**

---

## Confidence Score Tracking

Create a file to track confidence alongside accuracy:

```python
# validation_results.csv
Stage,CardType,Scan,Accuracy,AvgConfidence,MinConfidence,MaxConfidence,Coverage,Notes
1,Mastercard,3DES,99.2,0.87,0.75,0.99,100%,Perfect baseline
2,Mastercard,RSA,98.1,0.85,0.70,0.98,100%,Golden reference
2,Visa,RSA,95.2,0.81,0.68,0.97,100%,Expected cross-card degradation
3,Greenvisa,3DES,96.1,0.87,0.85,0.99,82%,Strict confidence - excellent
3,Greenvisa,3DES,91.2,0.78,0.70,0.99,98%,Full coverage - good
```

---

## Summary

Your validation strategy is **sound and progressive**:

1. **Stage 1:** Validate retrained models work on known cards (Mastercard 3DES)
2. **Stage 2:** Prove cross-card generalization works (Visa RSA using Mastercard-trained models)
3. **Stage 3:** Attack completely unknown card type (Greenvisa blind, evaluate confidence)

**Success = All stages pass with required accuracy + confidence correlation proven**

**Next:** Implement confidence scoring in your inference code, then run this checklist.
