# Blind Trace Confidence Scoring Framework

## Overview

This implementation provides a complete, production-ready framework for validating 3DES key predictions on **blind traces** (traces without ground truth data). The framework uses **consistency-based confidence scoring** to quantify prediction certainty without performing cryptographic verification.

**Key Design Principle:** Traces from the same card (identified by Track2) must yield the same cryptographic keys. Multi-trace agreement percentage serves as a practical confidence metric.

## Problem Context

During extensive testing, a fundamental data limitation was discovered:
- **Training Data Issue:** Only 4 S-Box outputs (2, 3, 5, 14) present in training dataset  
- **Ground Truth Conflict:** Ground truth key produces S-Box output 7 (not in training set)
- **Result:** Models consistently predict wrong keys with ~100% confidence

**Why Consistency Validation Works:**
- Same card → same key (cryptographic requirement)
- If multiple traces from one card consistently predict different keys → data quality issue
- If multiple traces predict same key → likely correct (even if data-limited)
- Consistency score indicates confidence level for deployment decisions

## Modules

### 1. `blind_trace_aggregation.py` (Reusable Utilities)
**Purpose:** Pure aggregation - works on any predictions DataFrame

**Key Functions:**
```python
aggregate_card_predictions(predictions_df, groupby_column='Track2', confidence_threshold=0.8)
  └─ Input: DataFrame with 3DES_KENC, 3DES_KMAC, 3DES_KDEK, Track2
  └─ Output: DataFrame with aggregated predictions + consistency metrics
  └─ Confidence levels: HIGH (≥80%), MEDIUM (60-80%), LOW (<60%)
  └─ Flagging: LOW confidence predictions marked for review

_compute_consistency(predictions: List[str]) → (ratio: float, most_common: str)
  └─ Computes agreement percentage and most frequent prediction

print_summary(aggregated_df)
  └─ Human-readable summary with HIGH/MEDIUM/LOW breakdown

save_aggregated_report(aggregated_df, output_dir)
  └─ Saves two CSV files:
     - blind_trace_aggregated_results.csv (all cards)
     - blind_trace_high_confidence_keys.csv (HIGH only)
```

### 2. `blind_trace_attack.py` (Production CLI)
**Purpose:** Post-processing entry point for inference pipeline

**Usage:**
```bash
python blind_trace_attack.py \
  --predictions_csv <path_to_predictions.csv> \
  --output_dir <output_directory> \
  --confidence_threshold 0.8  # Optional, default 0.8
```

**What it does:**
1. Load predictions from inference step
2. Group by card (Track2)
3. Aggregate predictions across traces
4. Score confidence by agreement %
5. Output aggregated results + high-confidence subset

### 3. `blind_trace_validator.py` (Full Orchestration)
**Purpose:** End-to-end validation orchestration (optional, for complex workflows)

**Key Classes:**
- `CardKeyEstimate` (dataclass): Typed result structure
- `BlindTraceValidator`: Full orchestration with CPA-level details

**Usage:**
```python
from src.blind_trace_validator import BlindTraceValidator

validator = BlindTraceValidator()
results = validator.validate_blind_traces(blind_traces_path, output_dir)
```

### 4. `test_blind_trace_scoring.py` (Test Suite)
**Purpose:** Validate framework with synthetic data

**What it tests:**
- ✅ Synthetic data generation (50 traces, 5 cards)
- ✅ Consistency computation accuracy
- ✅ Confidence level assignment (HIGH/MEDIUM/LOW)
- ✅ Output file generation
- ✅ Summary statistics and flagging

**Run tests:**
```bash
python test_blind_trace_scoring.py
```

## Data Flow

```
Blind Traces (trace_data, Track2, ATC)
        ↓
Existing Inference Pipeline (recover_3des_keys)
        ↓
Predictions CSV (3DES_KENC, 3DES_KMAC, 3DES_KDEK, Track2, )
        ↓
blind_trace_attack.py (post-processing)
        ↓
Aggregated Results + Confidence Scores
        ↓
Output CSVs:
  - blind_trace_aggregated_results.csv (all cards)
  - blind_trace_high_confidence_keys.csv (HIGH only)
```

## Output Format

### blind_trace_aggregated_results.csv
```
Card_ID | Num_Traces | Predicted_KENC  | Consistency_KENC | ... | Confidence_Level | Flagged_For_Review
card_001| 10         | AAAA111111111111| 100.0%          | ... | HIGH             | NO
card_002| 10         | AAAA222222222222| 80.0%           | ... | HIGH             | NO
card_003| 10         | AAAA333333333333| 50.0%           | ... | LOW              | YES
```

### blind_trace_high_confidence_keys.csv
Subset containing only HIGH confidence predictions (Confidence_Level == "HIGH")

## Example: Production Workflow

```python
import pandas as pd
from src.blind_trace_attack import run_blind_trace_attack

# After inference has created predictions.csv:
aggregated_df = run_blind_trace_attack(
    predictions_csv='Output/predictions_blind_traces.csv',
    output_dir='Output/blind_trace_validation',
    confidence_threshold=0.8
)

# Results:
# - blind_trace_aggregated_results.csv
# - blind_trace_high_confidence_keys.csv
#- Console summary with HIGH/MEDIUM/LOW breakdown
```

## Integration Points

### Option 1: Standalone Post-Processing (Recommended)
Run after inference completes:
```bash
# Step 1: Run inference
python pipeline-code/main.py --mode attack ...

# Step 2: Run confidence validation
python pipeline-code/blind_trace_attack.py \
  --predictions_csv Output/predictions.csv \
  --output_dir Output/blind_trace_validation
```

### Option 2: Python API
```python
from src.blind_trace_aggregation import aggregate_card_predictions

predictions_df = pd.read_csv('Output/predictions.csv')
aggregated = aggregate_card_predictions(predictions_df, confidence_threshold=0.8)
```

## Confidence Thresholds

Default thresholds (configurable):
- **HIGH:** ≥ 80% agreement across all three key types (KENC, KMAC, KDEK)
- **MEDIUM:** 60-80% agreement
- **LOW:** < 60% agreement (flagged for review)

*Note:* Agreement is computed per key type; final confidence uses average across types.

## Known Limitations

1. **Not cryptographic verification:** Consistency score ≠ correctness guarantee
   - If data is limited to specific ATC range, all traces may consistently predict same WRONG key
   - Consistency improves confidence reporting but cannot fix underlying data insufficiency

2. **Requires card grouping:** Must have Track2 (or equivalent card ID) in predictions

3. **Blind traces only:** Designed for traces without ground truth; not meant for traced-with-labels validation

## Test Results

All 5 tests passed successfully:
- ✅ Synthetic data (50 predictions, 5 cards)
- ✅ Consistency scoring (100%, 80%, 50%, 60%, 60%)
- ✅ Confidence assignment (HIGH → HIGH → LOW → MEDIUM → MEDIUM)
- ✅ Output generation (aggregated results + high-conf subset)
- ✅ Summary printing (formatted statistics)

**Output saved:** `Output/blind_trace_test/`

## Next Steps

1. **Run on actual blind trace predictions:**
   ```bash
   python blind_trace_attack.py \
     --predictions_csv <actual_predictions.csv> \
     --output_dir Output/blind_trace_actual
   ```

2. **Adjust confidence thresholds** if needed for your deployment criteria

3. **Review flagged predictions** (LOW confidence) for manual analysis

4. **Production deployment** using high-confidence keys only

## Architecture Decisions

**Why post-processing?**
- No model changes required
- No retraining needed
- Can be applied to predictions from any source
- Fast deployment (lightweight)
- Separates inference from validation concerns

**Why these three modules?**
1. **Aggregation** (reusable utilities) + **Attack** (CLI) = practical production use
2. **Validator** (orchestration) = optional for complex workflows requiring CPA details
3. **Tests** (validation suite) = confidence in framework correctness

## Troubleshooting

**Issue:** "Missing column: Track2"
- **Solution:** Predictions CSV must have Track2 column (or use card_id with manual mapping)

**Issue:** All predictions are LOW confidence
- **Likely cause:** Data limitation (limited ATC range in blind traces)
- **Mitigation:** Review sample predictions manually; consistency validation improves uncertainty reporting

**Issue:** No high-confidence keys generated  
- **Explanation:** All cards below 80% threshold
- **Action:** Adjust --confidence_threshold lower (e.g., 0.7) or analyze data quality issues

## References

- **Framework Architecture:** Consistency-based confidence scoring (no ground truth)
- **Validation Strategy:** Multi-trace agreement as confidence proxy
- **Output:** Separation of high/medium/low confidence for deployment decisions
- **Design Principle:** Quick deployment without model changes (pure post-processing)
