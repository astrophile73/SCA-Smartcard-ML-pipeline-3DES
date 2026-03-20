# 📋 COMPLETE DOCUMENTATION INDEX
## 3DES ML Pipeline - Root Cause Analysis & Fix Guide

**Date Created**: Today  
**Diagnostic Status**: ✅ ROOT CAUSE CONFIRMED  
**Recommended Action**: Start with Phase 1 (Week 1)

---

## 🚀 QUICK START (5 minutes)

1. **Read First**: [SUMMARY_AND_NEXT_STEPS.md](SUMMARY_AND_NEXT_STEPS.md)
   - One-page overview of the problem and solution
   - 3-week timeline
   - Success criteria

2. **Run This**: `python final_diagnostic.py`
   - Confirms the root cause
   - Shows evidence from your data
   - Takes 30 seconds

3. **Next Step**: Jump to [Phase 1 in IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#phase-1-verify-the-problem-today)

---

## 📚 DOCUMENTATION FILES

### **[1] SUMMARY_AND_NEXT_STEPS.md** ⭐ START HERE
- **Purpose**: Executive summary of the entire issue
- **Length**: 5 minutes to read
- **Contains**:
  - The problem in plain English
  - Why 58% accuracy is misleading
  - 3-week roadmap to fix
  - Quick-start commands

### **[2] ROOT_CAUSE_ANALYSIS.md** 🔍 TECHNICAL DETAILS
- **Purpose**: Deep technical explanation
- **Audience**: Developers who want to understand the issue deeply
- **Contains**:
  - Hamming Weight vs S-Box input (with examples)
  - Why current approach fails for CPA
  - Evidence from your specific files
  - Exact fixes needed

### **[3] CRITICAL_FINDINGS.md** 🚨 EVIDENCE
- **Purpose**: Presents the proof
- **Contains**:
  - Analysis of test predictions (all same key)
  - Label structure analysis
  - Hypotheses for why model defaults to one key
  - Required immediate fixes

### **[4] IMPLEMENTATION_GUIDE.md** 💻 CODE CHANGES
- **Purpose**: Step-by-step coding instructions
- **For**: Implementing the fixes
- **Contains**:
  - Phase-by-phase breakdown
  - Exact code to modify
  - Code snippets and examples
  - Testing procedures
  - Troubleshooting

### **[5] ACCURACY_IMPROVEMENT_ROADMAP.md** 📈 STRATEGIC PLAN
- **Purpose**: 4-week strategic improvement plan
- **Contains**:
  - Phase 1: Quick wins (optimization, architecture tweaks)
  - Phase 2: Data augmentation (3x more training data)
  - Phase 3: Multi-byte learning (ensemble methods)
  - Phase 4: Validation (cross-validation, testing)
  - Expected accuracy improvements per phase

### **[6] final_diagnostic.py** 🧪 EXECUTABLE
- **Purpose**: Run comprehensive diagnostic
- **Usage**: `python final_diagnostic.py`
- **Confirms**:
  - Labels are Hamming Weight (0-8), not S-Box input (0-255)
  - Model outputs same key for all test samples
  - This explains the low accuracy
  - Takes ~30 seconds

---

## 🎯 THE PROBLEM (One Sentence)

Your model predicts **Hamming Weight (0-8 bits)** instead of **S-Box input values (0-255)**, so it can't recover actual crypto keys—it just repeats the most common one for every test sample.

---

## ✅ THE SOLUTION (One Week)

1. Change label generation to extract 0-255 values (instead of 0-8) - **2 hours**
2. Update model output from 8 classes to 256 classes - **1 hour**
3. Retrain the model - **2-5 hours**
4. Verify: Should get 100+ different predictions instead of 1 - **30 minutes**

**Expected result**: Model that actually learns, predictions vary per trace, accuracy continues to improve with more data.

---

## 📊 TIMELINE

| Week | Task | Expected Accuracy |
|------|------|-------------------|
| Week 1 | Fix labels (0-255), Update model (256 classes), Retrain | 70-80% per byte |
| Week 2 | Data augmentation (3x), Optimized training | 75-85% per byte |
| Week 3 | Hyperparameter tuning, Cross-validation | 80-85%+ per byte |
| Week 4 | Validation, Testing, Documentation | Final report |

---

## 🔄 WORKFLOW

### Day 1: Diagnosis (Complete ✅)
```bash
# Already done for you - comprehensive diagnostic created
python final_diagnostic.py
# Result: Root cause confirmed
```

### Days 2-5: Fix Architecture
```bash
# Following IMPLEMENTATION_GUIDE.md Phase 1-2
# 1. Modify gen_labels_3des_fixed.py
# 2. Update model.py (256 classes)
# 3. Retrain with corrected data
python src/train.py --epochs=50 --batch-size=32
```

### Week 2: Data Augmentation
```bash
# Following IMPLEMENTATION_GUIDE.md Phase 3
# 1. Implement augmentation
# 2. Generate 3x more data
# 3. Retrain
python src/train.py --epochs=100 --batch-size=32 --augmented
```

### Week 3+: Optimization & Validation
```bash
# Following IMPLEMENTATION_GUIDE.md Phase 4
# 1. Hyperparameter tuning
# 2. Cross-validation
# 3. Final testing
```

---

## 📖 HOW TO USE THIS DOCUMENTATION

### If you want to understand the problem:
1. Read: [SUMMARY_AND_NEXT_STEPS.md](SUMMARY_AND_NEXT_STEPS.md)
2. View: Diagrams and pictures in the document
3. Run: `python final_diagnostic.py`

### If you want to fix it immediately:
1. Read: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) Phase 1
2. Make: Code changes listed in Phase 1
3. Run: Retraining command
4. Verify: Test predictions for diversity

### If you want deep technical understanding:
1. Read: [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md)
2. Follow: Each section's explanation
3. Check: References to [CRITICAL_FINDINGS.md](CRITICAL_FINDINGS.md) for evidence

### If you want a complete strategic plan:
1. Read: [ACCURACY_IMPROVEMENT_ROADMAP.md](ACCURACY_IMPROVEMENT_ROADMAP.md)
2. Follow: Week 1, Week 2, Week 3, Week 4 sections
3. Track: Progress with the provided checklists

---

## 🔍 KEY FILES IN YOUR PIPELINE

### Files That Need Changes
| File | Change | Priority |
|------|--------|----------|
| `pipeline-code/src/gen_labels_3des_fixed.py` | Extract 0-255 labels instead of HW | HIGH |
| `pipeline-code/src/model.py` | Output 256 classes not 8 | HIGH |
| `pipeline-code/src/train.py` | Add class weighting, use new labels | HIGH |
| `pipeline-code/src/inference_3des.py` | Output S-Box inputs not combined keys | HIGH |

### Files to Monitor
| File | Purpose |
|------|---------|
| `3des-pipeline/Processed/3des/Y_SBOX_INPUTS.npy` | New label file (to be created) |
| `3des-pipeline/Output/Report_After_Fix.csv` | Test report (should have >100 unique keys) |
| Training log files | Monitor loss/accuracy convergence |

---

## ✨ EXPECTED IMPROVEMENTS

### Before Fix (Current State)
```
❌ Labels: 2-4 unique values (Hamming Weight)
❌ Test output: 1 unique key across 1000 samples
❌ Accuracy: 58% (misleading metric)
❌ Predictions: All identical per trace
```

### After Week 1 Fix
```
✅ Labels: 256 unique values (S-Box inputs)
✅ Test output: 100-500 unique keys
✅ Per-byte accuracy: 70-80%
✅ Predictions: Different per trace
```

### After Week 2 Augmentation
```
✅ Per-byte accuracy: 75-85%
✅ Better coverage of S-Box space
✅ Increased prediction diversity
```

### After Week 3 Optimization
```
✅ Per-byte accuracy: 80-85%+
✅ Full-key recovery: 40%+
✅ Production-ready model
```

---

## ⚠️ COMMON ISSUES & SOLUTIONS

### Issue: "I don't understand the Hamming Weight problem"
**Solution**: Read the diagrams in [SUMMARY_AND_NEXT_STEPS.md](SUMMARY_AND_NEXT_STEPS.md) with pictures

### Issue: "Where do I start coding?"
**Solution**: Follow [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) Phase 1 step-by-step

### Issue: "How do I verify my changes work?"
**Solution**: Check section in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#phase-5-verify--measure-improvement-week-1)

### Issue: "Accuracy still low after fixes"
**Solution**: See troubleshooting section in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#troubleshooring)

---

## 🏗️ ARCHITECTURE OF SOLUTION

```
Problem:                           Solution:
┌──────────────────┐              ┌──────────────────┐
│ Model predicts   │              │ Model predicts   │
│ 0-8 (HW value)   │   ─────→     │ 0-255 (S-Box)    │
└──────────────────┘              └──────────────────┘
        │                                  │
        │                                  │
        └─→ Output: Same key             └─→ Output: Different keys
            (can't recover different keys   (can recover all keys)
             properly, defaults to mode)    with varying accuracy
```

---

## 📞 QUICK REFERENCE COMMANDS

```bash
# Check the current state
python final_diagnostic.py

# After fixing labels and model:
python pipeline-code/src/train.py --epochs=50 --batch-size=32

# Test predictions on new model:
python pipeline-code/src/inference_3des.py --verify

# Check if predictions are now diverse:
python -c "
import pandas as pd
df = pd.read_csv('3des-pipeline/Output/Report_After_Fix.csv')
print(f'Unique predictions: {df[\"3DES_KENC\"].nunique()}')
"

# View improvement:
python -c "
import os
print('Current accuracy reports:')
for f in os.listdir('3des-pipeline/Output'):
    if 'Report' in f:
        print(f'  {f}')
"
```

---

## 📊 SUCCESS METRICS

After implementing these fixes, track these metrics:

| Metric | Week 0 (Now) | Week 1 (Target) | Week 2 (Target) | Week 3 (Target) |
|--------|--------------|-----------------|-----------------|-----------------|
| Unique keys in test set | 1 | 100+ | 200+ | 300+ |
| Per-byte accuracy | N/A | 70-80% | 75-85% | 80-85%+ |
| Predictions vary per trace | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Model learning | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |

---

## 🎓 LEARNING PATH

If this is your first time looking at this:

1. **Beginner**: Start with [SUMMARY_AND_NEXT_STEPS.md](SUMMARY_AND_NEXT_STEPS.md)
   - Skip technical details
   - Focus on "What's wrong" and "How to fix it"
   - Read the one-sentence problems/solutions

2. **Intermediate**: Add [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md)
   - Understand why Hamming Weight doesn't work
   - See the examples with multiple keys having same HW
   - Understand why CPA needs exact values

3. **Advanced**: Use [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
   - Make actual code changes
   - Follow the phase-by-phase breakdown
   - Implement augmentation and optimization

4. **Expert**: Reference [ACCURACY_IMPROVEMENT_ROADMAP.md](ACCURACY_IMPROVEMENT_ROADMAP.md)
   - Design the 4-week strategic plan
   - Optimize beyond the baseline
   - Implement advanced techniques

---

## 🎯 YOUR NEXT ACTION

**RIGHT NOW** (5 minutes):
```
1. Read: SUMMARY_AND_NEXT_STEPS.md (5 min)
2. Run:  python final_diagnostic.py (30 sec)
```

**TODAY** (If you want to start coding):
```
1. Read: IMPLEMENTATION_GUIDE.md Phase 1
2. Make: Changes to gen_labels_3des_fixed.py
3. Generate: New training labels
4. Run: Quick test with 20 epochs
```

**THIS WEEK**:
```
1. Complete: All Phase 1 changes
2. Verify: Output has >100 different predicted keys
3. Measure: Per-byte accuracy improvement
```

---

## 📞 NEED HELP?

1. **Can't understand the problem?** → Read [SUMMARY_AND_NEXT_STEPS.md](SUMMARY_AND_NEXT_STEPS.md)
2. **Don't know where to start coding?** → Follow [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) Phase 1
3. **Want technical details?** → See [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md)
4. **Looking for evidence?** → Check [CRITICAL_FINDINGS.md](CRITICAL_FINDINGS.md)
5. **Want a full plan?** → Use [ACCURACY_IMPROVEMENT_ROADMAP.md](ACCURACY_IMPROVEMENT_ROADMAP.md)

---

**Status**: ✅ Diagnosis Complete | Ready for Implementation | Expected 2-3 week fix

