# 3DES PIPELINE - RESOLUTION SUMMARY & ACTION PLAN

## Problem Statement
**"Recovered keys are wrong"** - Models producing incorrect 3DES keys

### Root Cause Analysis
**✓ IDENTIFIED: Only 43.9% of traces had valid 3DES keys**
- 20,000 traces with valid keys (T_DES_KENC/KMAC/KDEK columns populated)
- 25,606 traces with missing/invalid keys (T_DES_KENC/KMAC/KDEK = NaN or -1)
- **Models trained on corrupted dataset = corrupted outputs**

### Solutions Implemented
**3 Critical Bugs Fixed in `ingest.py`:**
1. `_normalize_key_hex()` now handles numpy scalar types (line 138-157)
2. Context manager wrapper for .npy file loading (line 220-263)
3. Safe key extraction from NPZ/CSV arrays (line 325-363)

**External Label Integration:**
- KALKI TEST CARD.xlsx provides 100% key coverage (45,606/45,606 traces)
- Mastercard key: `9E15204313F...`
- Visa key: `23152081ECF...`

---

## Answers to Your 3 Questions

### Q1: Do we need to re-train the models?
**✓ YES - MANDATORY**

The previous models were trained on only 43.9% valid labels. Retraining with 100% key coverage is non-negotiable.

```bash
python retrain_with_full_coverage.py --dry_run  # First check what will run
python retrain_with_full_coverage.py --backup   # Back up old models before retraining
```

**Timeline: ~1-2 hours**

---

### Q2: Are there issues in preprocessing/post-attack?

**Status Summary:**
| Component | Status | Issue | Next Action |
|-----------|--------|-------|------------|
| **Preprocessing** | ⚠️ NEEDS RETRAIN | Was trained with 43.9% invalid labels | Run retrain script |
| **Feature Extraction** | ✓ OK | Key-agnostic (uses POI indices only) | No changes needed |
| **Label Computation** | ✓ OK | Now works with 100% valid keys | Will be fixed by retrain |
| **Training** | ✓ READY | Code is correct, just needs complete data | Run retrain script |
| **Inference/Attack** | ✓ OK | Can handle blind traces (no labels needed) | No changes needed |
| **Post-Attack** | ✓ OK | Key recovery logic is deterministic | No issues found |

**Bottom Line:** Preprocessing will be fixed by retraining. Post-attack components have no issues.

---

### Q3: How to handle blind traces & ensure generalization?

**Blind Traces (Unknown keys, no labels):**
- ✓ System CAN handle blind traces
- Blind traces don't need labels during inference
- Only limitations: Model quality depends on training completeness

**Ensure Generalization:** ⚠️ CRITICAL - UNPROVEN AND HIGH RISK

```
Current Risk:
└─ Models trained on Mastercard key (9E15...) + Visa key (2315...)
   └─ Will they work on OTHER Mastercard keys? UNKNOWN
   └─ Will they work on OTHER Visa keys? UNKNOWN
   └─ Features are power-based (generalizable)
   └─ BUT labels are key-specific (over-fitting risk)
```

**Validation Required:**
```bash
python validate_generalization.py \
  --train_profile Mastercard \
  --test_profile Visa \
  --accuracy_threshold 0.90  # Minimum for deployment
```

This will:
1. Train models on Mastercard traces only
2. Test blind on Visa traces
3. Report if generalization achieved

**Expected Results:**
- ✓ **> 90% accuracy** → True generalization, safe for production
- ⚠️ **70-90% accuracy** → Partial generalization, needs more training data
- ✗ **< 70% accuracy** → No generalization, expand KALKI file with more keys

---

## Complete Action Plan

### **Week 1: Fix Data Coverage & Retrain**

**Step 1: Retrain with Full Coverage** (Today)
```bash
# Backup old models first
python retrain_with_full_coverage.py --dry_run    # Verify what will happen
python retrain_with_full_coverage.py --backup     # Backup + retrain
# Expected time: 1-2 hours
# Output: New trained models in 3des-pipeline/models/3des/
```

**Step 2: Initial Validation** (Today)
```bash
python pipeline-code/main.py \
  --action attack \
  --attack_type inference_3des \
  --models_dir 3des-pipeline/models/3des
# Check if recovered keys are correct on test Mastercard + Visa traces
```

### **Week 1-2: Critical Generalization Testing**

**Step 3: Cross-Card Generalization** (This week - CRITICAL)
```bash
python validate_generalization.py \
  --train_profile Mastercard \
  --test_profile Visa \
  --accuracy_threshold 0.90
```

**Interpreting Results:**
- ✓ **PASS (> 90%)**: Proceed to production
- ⚠️ **PARTIAL (70-90%)**: Expand KALKI, retrain
- ✗ **FAIL (< 70%)**: Architecture needs review

### **Week 2-3: Production Hardening** (If generalization passes)

**Step 4: Add Confidence Scoring**
```python
# Update inference_3des.py to return:
# (recovered_keys, confidence_scores)
# Only accept keys with confidence > threshold
```

**Step 5: Production Deployment**
```
- Deploy retrained models to production
- Set up monitoring for key recovery accuracy
- Document model limitations (trained on 2 card types)
- Create runbook for handling low-confidence predictions
```

### **Week 3+: Optional - Model Improvement** (If generalization fails)

**Step 6a: Expand KALKI File**
```
Get access to more transaction records:
- 5-10 different Mastercard keys  
- 5-10 different Visa keys
- Create expanded_KALKI.xlsx
```

**Step 6b: Retrain with Diversity**
```bash
python retrain_with_full_coverage.py \
  --label_map expanded_KALKI.xlsx \
  --epochs 300  # More epochs, more data = better generalization
```

**Step 6c: Re-validate**
```bash
python validate_generalization.py --accuracy_threshold 0.95
```

---

## Key Files & Their Roles

### **Created/Modified:**
- `retrain_with_full_coverage.py` - **Main retraining orchestrator**
- `validate_generalization.py` - **Critical generalization validator**
- `RETRAINING_DECISION.md` - **Detailed technical decisions**
- `GENERALIZATION_ANALYSIS.py` - **Identification of generalization risks**
- `pipeline-code/src/ingest.py` - **Fixed (3 critical bugs)**

### **Important References:**
- [KALKI TEST CARD.xlsx](KALKI%20TEST%20CARD.xlsx) - External label source
- [RETRAINING_DECISION.md](RETRAINING_DECISION.md) - Technical deep-dive
- [pipeline-code/main.py](pipeline-code/main.py) - Main pipeline orchestrator (lines 123-124 for label flags)

---

## Risk Assessment

### **Risk 1: Deploy Model Trained on 43.9% Valid Labels** (CRITICAL)
- **Probability**: High if retraining skipped
- **Impact**: All recovered keys wrong
- **Mitigation**: ✓ Run retrain script

### **Risk 2: Model Only Works on KALKI Cards** (CRITICAL)  
- **Probability**: High (only 2 card types in training)
- **Impact**: Deploy fails in field on new card types
- **Mitigation**: ✓ Run generalization validator before production

### **Risk 3: No Confidence Scoring on Blind Traces** (HIGH)
- **Probability**: Medium
- **Impact**: Return wrong keys with no way to detect errors
- **Mitigation**: ✓ Add confidence threshold to inference (Week 2 task)

---

## Technical Insights

### **Why Models Might Not Generalize**

```
Training Process:
┌─ Traces (Mastercard + Visa) 
│  └─ Extract POI power values (key-agnostic) ✓
│  └─ Compute S-box labels using specific keys (key-dependent) ✗
│  
└─ Train model: power_values → s_box_outputs

The Problem:
- If Mastercard key is 9E15... and Visa key is 2315...
- S-box outputs are different for SAME power patterns
- Model learns: "2315... key produces THESE s-box outputs"
- When testing different Visa key: outputs change → confusion

The Solution:
- Train with MULTIPLE different keys per card type (5-10 each)
- Forces model to learn: "Same S-box always produces same pattern"
- True generalization via diverse training
```

### **Why Blind Traces Work**

```
Inference (Attack):
1. Blind trace (unknown key, no labels) comes in
2. Model predicts S-box outputs (learned pattern)
3. Reconstruct key that would produce those outputs
4. Return recovered key

This works IF:
- Models learned true S-box leakage (generalizable)
- Not just memorized specific keys (non-generalizable)
```

---

## Success Criteria

### **Model is ready for production when:**

✓ Retraining complete with 100% key coverage  
✓ Generalization validator passes (> 90% accuracy cross-card)  
✓ Confidence scoring implemented  
✓ Production monitoring in place  
✓ Documentation of model limitations  

### **Model is NOT ready if:**
✗ Generalization < 85% on unseen card types  
✗ No confidence threshold on blind predictions  
✗ Only one card type in training data  

---

## Quick Reference: Shell Commands

```bash
# 1. DRY RUN (see what will happen)
python retrain_with_full_coverage.py --dry_run

# 2. BACKUP + RETRAIN  
python retrain_with_full_coverage.py --backup

# 3. TEST GENERALIZATION (Mastercard-trained on Visa)
python validate_generalization.py --train_profile Mastercard --test_profile Visa

# 4. TEST GENERALIZATION (Visa-trained on Mastercard)
python validate_generalization.py --train_profile Visa --test_profile Mastercard

# 5. ENHANCED TRAINING (with more epochs for complex data)
python retrain_with_full_coverage.py --epochs 300 --batch_size 16
```

---

## Support & Next Steps

**Immediate (Next 2 hours):**
- Review this document
- Run `retrain_with_full_coverage.py --dry_run` to understand pipeline
- Ensure KALKI TEST CARD.xlsx is in workspace

**This Week:**
- Execute full retrain
- Run generalization validator
- Document results

**Before Production:**
- Confirm generalization test passes
- Implement confidence scoring
- Set up monitoring

---

**Document Created:** [RESOLUTION SUMMARY]  
**Status:** Actionable (all scripts ready to execute)  
**Next Action:** Run retraining script  
**Critical Path:** Retraining → Generalization Test → Production Deployment
