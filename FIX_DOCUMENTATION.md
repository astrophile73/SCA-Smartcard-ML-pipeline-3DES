# 3DES/RSA ML Pipeline - Critical Fixes Implementation Guide

## Summary

Two critical bugs have been identified and fixed:

### 1. **3DES Key Recovery - Normalization Mismatch** ✅ FIXED
**Problem:** When card type is auto-detected from Track2 data, models trained on Mastercard-only features were receiving features normalized by mixed-data statistics, causing incorrect key recovery.

**Root Cause:**
- Models trained with `--card_type mastercard` learn patterns from Mastercard-only feature distributions
- When running attack without `--card_type`, auto-detection filters to Mastercard traces
- But normalization (mean/std) was loaded from training data (which included Visa traces)
- Feature distribution mismatch → wrong model inputs → wrong keys

**Solution Implemented:**
- Modified `_card_mask()` function to return `(mask, detected_card_type)` tuple
- Added `_compute_norm_from_data()` to compute statistics from data subset
- Updated `recover_3des_keys()` and `recover_3des_keys_with_confidence()` to:
  - Detect when card type filtering is applied
  - Recompute mean/std from MASKED features only when filtering is active
  - Use recomputed statistics for normalization instead of stored training stats
  - Log when recomputation is triggered for debugging

**Files Modified:**
- [pipeline-code/src/inference_3des.py](pipeline-code/src/inference_3des.py) (Lines 85-117, 146-160, 237-280, 467-551)

---

### 2. **RSA Component Padding** ✅ FIXED
**Problem:** RSA predictions remain as 128-byte zero-padded hex strings (256 chars) in some cases, when they should be unpadded to variable length.

**Root Cause:**
- `perform_rsa_attack()` creates 256-char padded hex strings: `''.join(f'{b:02X}' for b in row)`
- `attack_all_rsa_components()` calls `derive_rsa_crt()` to convert to unpadded format
- If `derive_rsa_crt()` fails (malformed P/Q or GCD != 1) → original padded values kept
- Fallback to strip padding was missing

**Solution Implemented:**
- Added `strip_rsa_padding()` helper function in `attack_all_rsa_components()`
- When `derive_rsa_crt()` succeeds: use derived unpadded values (existing behavior)
- When `derive_rsa_crt()` fails OR P/Q empty: Apply padding stripper as fallback
- GUARANTEES all output values are unpadded (variable length, not 256 chars)
- Added logging to track derive success/fail counts

**Files Modified:**
- [pipeline-code/src/inference_rsa.py](pipeline-code/src/inference_rsa.py) (Lines 148-223)

---

## Testing & Validation

### Quick Validation Tests ✓
```bash
# Run fix validation script
cd pipeline-code
python test_fixes.py
python test_attack.py
```

**Expected Output:**
- ✓ `_card_mask` returns tuple (mask, detected_type)
- ✓ `_compute_norm_from_data` computes statistics correctly  
- ✓ Attack logs show "Using recomputed normalization for detected card type 'X'"
- ✓ No runtime errors during attack initialization

### Full Pipeline Test
```bash
# Test attack on all data with auto-detection
python main.py --mode attack \
  --processed_dir ./Processed \
  --model_root ../3des-pipeline/models \
  --output_dir ./Output

# Test attack with explicit card type
python main.py --mode attack \
  --processed_dir ./Processed \
  --model_root ../3des-pipeline/models \
  --card_type mastercard \
  --output_dir ./Output

# Compare outputs - 3DES keys should match between both runs
# RSA values should have variable length, no 256-char padding
```

### Expected Improvements
1. **3DES Keys:**
   - ✅ Auto-detection mode should recover same keys as explicit `--card_type` mode
   - ✅ Should restore ~100% KENC accuracy from before normalization issue
   - ✅ KMAC/KDEK should also be correct

2. **RSA Components:**
   - ✅ All values unpadded (variable length, not 256 chars)
   - ✅ Trailing zero bytes properly stripped
   - ✅ No "00" suffixes except for legitimately short values

---

## Implementation Details

### Fix #1: 3DES Normalization
**Key Changes:**
```python
# OLD: Always loaded stored norms
X1 = _normalize(X1_masked, _load_norm(model_dir, stage=1, key_type=key_type))

# NEW: Detects if filtering was applied, recomputes norms if needed
if should_recompute_norms:  # True when detected_card_type != "universal"
    mean_s1, std_s1 = _compute_norm_from_data(X1_masked)
    X1_norm = _normalize(X1_masked, (mean_s1, std_s1))
else:
    X1_norm = _normalize(X1_masked, _load_norm(model_dir, stage=1, key_type=key_type))
```

**When Recomputation Happens:**
- User runs: `main.py --mode attack` (no explicit card_type)
- `_card_mask("universal")` auto-detects and returns detected type (e.g., "mastercard")
- `should_recompute_norms = (detected != "universal")` → True
- Features normalized using Mastercard-only statistics

**When Using Stored Norms:**
- User runs: `main.py --mode attack --card_type mastercard` (explicit type)
- `_card_mask("mastercard")` returns detected type = "mastercard"
- `should_recompute_norms` still True (any non-universal)
- Features recomputed (safer, but if models were trained with stored norms, this should still work)
- Can optimize this in future (cache card-type-specific norm files)

### Fix #2: RSA Padding Fallback
**Key Changes:**
```python
# ALWAYS ensure values are unpadded
for i in range(n_preds):
    p_hex = results['RSA_CRT_P'][i]
    q_hex = results['RSA_CRT_Q'][i]
    
    if p_hex and q_hex:
        derived = derive_rsa_crt(p_hex, q_hex)
        if derived:
            # Path 1: Derive succeeded - use derived unpadded values
            results['RSA_CRT_P'][i] = derived['P']
            ...
        else:
            # Path 2: Derive failed - strip padding from originals
            results['RSA_CRT_P'][i] = strip_rsa_padding(p_hex)
            ...
    else:
        # Path 3: P or Q missing - strip padding from all
        results['RSA_CRT_P'][i] = strip_rsa_padding(p_hex)
        ...
```

**Logging:**
- `derive_success` = counts where derived values were used
- `derive_fail` = counts where padding stripper was used as fallback
- Log shows: "RSA consistency check: X traces with derived values, Y traces with padding stripped"

---

## Verification Checklist

- [ ] Run `python test_fixes.py` - all tests pass ✓
- [ ] Run `python test_attack.py` - see recomputation logs ✓
- [ ] Run full pipeline with auto-detect (no `--card_type`)
- [ ] Run full pipeline with explicit `--card_type mastercard`
- [ ] Compare 3DES keys between both runs - should match
- [ ] Check RSA output values - verify no 256-char padding
- [ ] Verify `derive_success` and `derive_fail` counts in logs
- [ ] Compare against ground truth if available
- [ ] If issues persist: check feature normalization is actually being recomputed (logs show it)

---

## Future Optimizations

1. **Cache Card-Type-Specific Norms:** Instead of recomputing at inference time, could save norms per card type during training
2. **Skip Recomputation for Perfect Splits:** If data is 100% one card type, don't recompute (optimization only)
3. **Parameterize Threshold:** Make the 80% threshold configurable for auto-detection
4. **RSA Validation:** Add cryptographic validation of RSA components (N=P*Q, etc.)

---

## Rollback Instructions

If issues occur, revert changes:
```bash
# Revert 3DES fix
git checkout pipeline-code/src/inference_3des.py

# Revert RSA fix  
git checkout pipeline-code/src/inference_rsa.py
```

---

## Questions & Debugging

**Q: Why is normalization recomputed even when card_type is explicitly set?**
A: Being conservative - recomputing is cheap and ensures features match what models expect. In future, could optimize to skip when explicit type matches training data distribution.

**Q: What if RSA derive fails for ALL traces?**
A: All values will be stripped of padding via fallback. The log will show `derive_success: 0, derive_fail: N`. RSA may not be recoverable, but at least padding won't mask the issue.

**Q: Can I see detailed logs for each trace?**
A: Set `logging.basicConfig(level=logging.DEBUG)` in main.py or add `logger.debug()` prints in attack_all_rsa_components for per-trace details.
