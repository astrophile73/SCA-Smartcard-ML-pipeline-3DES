# 3DES PIPELINE FIX - QUICK REFERENCE

## The Problem
"Recovered keys are wrong" → **43.9% of traces had missing/invalid keys**

## The Solution in 3 Steps

### Step 1️⃣: UNDERSTAND
```
✓ Only 20,000/45,606 traces had valid 3DES keys
✓ 25,606 traces had label=-1 (missing/invalid)
✓ Models trained on corrupted data = wrong outputs
✓ Fixed 3 bugs in ingest.py
✓ External labels (KALKI file) provide 100% coverage
```

### Step 2️⃣: RETRAIN (Mandatory)
```bash
python retrain_with_full_coverage.py --dry_run    # See what happens
python retrain_with_full_coverage.py --backup     # Backup & retrain
# Takes ~1-2 hours
```

### Step 3️⃣: VALIDATE GENERALIZATION (Critical)
```bash
python validate_generalization.py --train_profile Mastercard --test_profile Visa
# If accuracy > 90%: Models generalize! ✓
# If accuracy < 85%: Cannot deploy yet ✗
```

---

## Answers to Your 3 Questions

| Question | Answer | Action |
|----------|--------|--------|
| **Do we need to re-train?** | ✓ YES - Mandatory | Run `retrain_with_full_coverage.py` |
| **Preprocessing/post-attack issues?** | ✓ Will be fixed by retrain | No manual fixes needed |
| **Blind traces & generalization?** | ⚠️ Unknown - needs testing | Run `validate_generalization.py` |

---

## Files to Run

### For Retraining:
```bash
python retrain_with_full_coverage.py --backup --epochs 200
```
**Output:** New trained models in `3des-pipeline/models/3des/`  
**Time:** 1-2 hours  
**What it does:** Uses 100% valid keys to retrain all ensemble models  

### For Validation:
```bash
python validate_generalization.py --train_profile Mastercard --test_profile Visa
```
**Output:** Generalization accuracy report  
**Time:** 45-90 minutes  
**What it does:** Tests if Mastercard-trained models work on Visa (blind)  

---

## Key Insights

### **Why Generalization is Critical**
```
Current training:
  └─ 1 Mastercard key (9E15...)
  └─ 1 Visa key (2315...)
  
Will models work on OTHER keys? UNKNOWN!

Generalization test answers: "Can Mastercard models recover Visa keys?"
  - YES (>90%) → True generalization achieved
  - NO (<85%) → Models overfit to specific keys
```

### **What Changed**
```
Before:  Labels for 20,000 traces | Invalid for 25,606 | Bad training
After:   Labels for 45,606 traces | 100% valid        | Good training!
```

### **Blind Traces**
```
Can inference work without labels?
  - Input: Blind power trace (unknown key)
  - Models predict S-box outputs (learned pattern)
  - Recover key from outputs
  - ✓ YES - works if models generalize
```

---

## Quick Decision Tree

```
START
  ↓
[Run retrain script]
  ↓
Models retrained with 100% keys?
  ├─ NO  → Check prerequisites, run with --dry_run first
  └─ YES ↓
       [Run generalization test]
         ↓
       Generalization > 90%?
         ├─ YES → ✓ READY FOR PRODUCTION
         │        └─ Add confidence scoring
         │        └─ Deploy
         │        └─ Monitor
         │
         └─ NO  → ⚠️ NEEDS MORE TRAINING DATA
                  └─ Expand KALKI file (5-10 keys per card)
                  └─ Retrain with --epochs 300
                  └─ Re-run validation
```

---

## Risk Summary

| Risk | Severity | Status | Mitigation |
|------|----------|--------|-----------|
| Deploy model trained on 43.9% valid labels | CRITICAL | IDENTIFIED | ✓ Run retrain |
| Model only works on KALKI test cards | CRITICAL | UNTESTED | ✓ Run generalization test |
| No confidence scoring on blind predictions | HIGH | TODO | Add threshold after retrain |

---

## Timeline

```
Today:           Read this doc, run --dry_run
This week:       Full retrain (1-2 hrs) + generalization test (45-90 min)
Week 2:          Add confidence scoring if needed
Week 2-3:        Production deployment (if generalization passes)
Week 3+:         Optional: Expand KALKI & retrain for better diversity
```

---

## Files Created to Help You

| File | Purpose | Run When |
|------|---------|----------|
| **retrain_with_full_coverage.py** | Main script to retrain models | Before models go to production |
| **validate_generalization.py** | Test cross-card generalization | After retraining, before deployment |
| **RESOLUTION_SUMMARY.md** | Complete technical document | For detailed understanding |
| **RETRAINING_DECISION.md** | Detailed decision logic | For technical context |
| **GENERALIZATION_ANALYSIS.py** | Analysis of generalization risks | To understand the risks |

---

## One-Command Cheatsheet

```bash
# See what retraining will do (safe, no changes)
python retrain_with_full_coverage.py --dry_run

# Actually retrain (backs up old models first)
python retrain_with_full_coverage.py --backup

# Test if Mastercard models generalize to Visa
python validate_generalization.py --train_profile Mastercard --test_profile Visa

# Test the other way (Visa → Mastercard)
python validate_generalization.py --train_profile Visa --test_profile Mastercard
```

---

## Success Indicators

✓ **When you know you're good to go:**
- Retrain script completes without errors
- Generalization test shows > 90% accuracy
- Models produce same keys repeatedly (consistency)
- Different card types both show high accuracy

✗ **When you need to dig deeper:**
- Generalization < 85% on cross-card test
- Models fail on any card type beyond KALKI
- No confidence threshold implemented
- Warning: Don't deploy until generalization is fixed!

---

## Help & Support

**Questions about retraining?**
→ See `retrain_with_full_coverage.py --help`

**Questions about generalization?**
→ See `validate_generalization.py --help`

**Want full technical details?**
→ Read `RESOLUTION_SUMMARY.md`

**Need architecture deep-dive?**
→ Read `RETRAINING_DECISION.md`

**Curious about the risks?**
→ Read `GENERALIZATION_ANALYSIS.py` output

---

**Status:** Ready to execute  
**Next Action:** `python retrain_with_full_coverage.py --dry_run`  
**Estimated Time to Production:** 1 week
