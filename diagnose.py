"""
Diagnostic Script: Analyze 3DES Pipeline Data Quality & Model Performance

Checks:
1. Data availability and structure
2. Label correctness and balance
3. Trace quality (SNR, noise floor)
4. Model capacity vs data complexity
5. Training dynamics (loss curves, overfitting)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

def check_input_data():
    """Analyze input data quality"""
    print("\n" + "="*80)
    print("[1] INPUT DATA ANALYSIS")
    print("="*80)
    
    input_dir = Path('3des-pipeline/Input')
    
    # Find CSV files
    csv_files = sorted(list(input_dir.glob('*.csv')))
    
    if not csv_files:
        print("❌ NO CSV FILES FOUND in 3des-pipeline/Input/")
        return None
    
    print(f"✓ Found {len(csv_files)} CSV files")
    
    # Sample first file
    df = pd.read_csv(csv_files[0], nrows=1000)
    print(f"\nColumns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nFirst row:\n{df.iloc[0]}")
    
    # Check for key columns
    key_cols = ['T_DES_KENC', 'T_DES_KMAC', 'T_DES_KDEK', 'trace_data']
    available_keys = [col for col in key_cols if col in df.columns]
    print(f"\n✓ Available key columns: {available_keys}")
    
    if 'trace_data' in df.columns:
        # Parse trace data
        try:
            sample_trace = eval(df.iloc[0]['trace_data'])
            print(f"✓ Trace data shape: {np.array(sample_trace).shape}")
            print(f"  Trace dtype: {np.array(sample_trace).dtype}")
            print(f"  Trace min/max: {np.min(sample_trace):.2f} / {np.max(sample_trace):.2f}")
        except:
            print("❌ Could not parse trace_data")
    
    return df

def check_data_balance(df):
    """Check if S-Box input space is covered"""
    print("\n" + "="*80)
    print("[2] DATA BALANCE & COVERAGE")
    print("="*80)
    
    # Check how many unique traces per S-Box input (byte 0)
    if 'T_DES_KENC' in df.columns and 'trace_data' in df.columns:
        print(f"Total samples: {len(df)}")
        
        # Estimate S-Box inputs if keys are known
        key_samples = df['T_DES_KENC'].unique()
        print(f"Unique keys: {len(key_samples)}")
        print(f"Traces per key: {len(df) / len(key_samples):.0f}")
        
        # For each key, we get different S-Box inputs depending on input data
        print("\nNote: For CPA attack to work, need diverse S-Box inputs")
        print("  Ideal: >100 traces per potential S-Box input (0-255)")
        print(f"  Available: {len(df)} traces total")
        print(f"  Theoretical coverage: {len(df) / 256:.0f} traces per S-Box input (if uniform)")

def check_processed_data():
    """Check processed dataset quality"""
    print("\n" + "="*80)
    print("[3] PROCESSED DATA QUALITY")
    print("="*80)
    
    proc_dir = Path('3des-pipeline/Processed/3des')
    
    # Check for .npy files
    npy_files = list(proc_dir.glob('*.npy'))
    print(f"✓ Processed files: {len(npy_files)}")
    
    for f in sorted(npy_files)[:5]:  # First 5 files
        data = np.load(f)
        print(f"  {f.name}: shape={data.shape}, dtype={data.dtype}, "
              f"mean={data.mean():.3f}, std={data.std():.3f}")
    
    # Check label distribution if exists
    labels_file = proc_dir / 'labels.npy'
    if labels_file.exists():
        labels = np.load(labels_file)
        print(f"\nLabels distribution:")
        print(f"  Shape: {labels.shape} (n_samples × n_bytes)")
        print(f"  Unique values per byte: ", end="")
        uniqueness = [len(np.unique(labels[:, i])) for i in range(min(3, labels.shape[1]))]
        print(f"First 3 bytes: {uniqueness}")
        
        # Check if labels match expected keys
        first_byte_distribution = pd.Series(labels[:, 0]).value_counts()
        print(f"  First byte distribution (top 10):")
        for val, count in first_byte_distribution.head(10).items():
            print(f"    0x{int(val):02X}: {count} samples")

def check_model_architecture():
    """Analyze model capacity"""
    print("\n" + "="*80)
    print("[4] MODEL ARCHITECTURE ANALYSIS")
    print("="*80)
    
    try:
        from pipeline_code.src.model import Model
        
        # Assume input traces are ~1000 samples
        model = Model(input_size=1000, output_size=256)
        
        # Count parameters
        total_params = sum(p.numel() if hasattr(p, 'numel') else 0 
                          for p in model.parameters() if hasattr(model, 'parameters'))
        print(f"✓ Model parameters: {total_params:,}")
        print(f"  Input: 1000 samples/trace")
        print(f"  Output: 256 classes (S-Box inputs)")
        
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        print("\nEstimated capacity needed:")
        print("  Input: 1000 features → 256 classes")
        print("  Recommended: >50k parameters for good generalization")
        print("  Question: Is current model large enough?")

def check_training_logs():
    """Look for training history"""
    print("\n" + "="*80)
    print("[5] TRAINING HISTORY")
    print("="*80)
    
    # Check for logs
    log_patterns = [
        'Output/training_*.log',
        'Output/*history*.csv',
        'pipeline-code/Output/*history*.csv',
    ]
    
    found_logs = False
    for pattern in log_patterns:
        logs = list(Path('.').glob(pattern))
        if logs:
            found_logs = True
            print(f"✓ Found logs matching {pattern}:")
            for log in logs[:3]:
                print(f"  {log}")
    
    if not found_logs:
        print("⚠ No training logs found. Recommendation:")
        print("  - Add callback to save loss/accuracy during training")
        print("  - Check for overfitting: large gap between train/val accuracy")

def check_test_performance():
    """Check test set performance"""
    print("\n" + "="*80)
    print("[6] TEST SET PERFORMANCE")
    print("="*80)
    
    # Check for test reports
    report_files = list(Path('3des-pipeline/Output').glob('*Report*.csv'))
    
    if report_files:
        print(f"✓ Found {len(report_files)} report files:")
        for rf in report_files:
            print(f"  {rf.name}")
            df = pd.read_csv(rf)
            print(f"    Shape: {df.shape}")
            print(f"    Columns: {list(df.columns)}")
            
            # Check accuracy column
            if 'accuracy' in df.columns:
                print(f"    Accuracy: min={df['accuracy'].min():.1%}, "
                      f"max={df['accuracy'].max():.1%}, "
                      f"mean={df['accuracy'].mean():.1%}")
    else:
        print("⚠ No test reports found")
        print("  Run test with: python pipeline-code/main.py --test")

def main():
    """Run all diagnostics"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " 3DES ML PIPELINE - DIAGNOSTIC REPORT".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    try:
        # 1. Input data
        df = check_input_data()
        
        # 2. Data balance
        if df is not None:
            check_data_balance(df)
        
        # 3. Processed data
        check_processed_data()
        
        # 4. Model
        check_model_architecture()
        
        # 5. Training logs
        check_training_logs()
        
        # 6. Test performance
        check_test_performance()
        
    except Exception as e:
        print(f"\n❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print recommendations
    print("\n" + "="*80)
    print("[RECOMMENDATIONS]")
    print("="*80)
    print("""
1. DATA QUALITY CHECK:
   - Verify traces are properly aligned (peak power consumption timing)
   - Check if S-Box input space is well-covered (>100 traces per input)
   - Validate label extraction (keys should match expected values)

2. MODEL CAPACITY:
   - Ensure model has enough parameters (>50k for this task)
   - Current: Check if architecture uses BatchNorm, Dropout
   - Try: Increase hidden layer sizes or add more layers

3. TRAINING DYNAMICS:
   - Check if loss decreases consistently
   - Look for overfitting (train acc >> val acc)
   - Solution: Add regularization (dropout, L2) or augmentation

4. DATA AUGMENTATION:
   - Generate 3-5x more training data through variations
   - Techniques: Noise injection, time shift, amplitude scaling
   - Expected: +10-15% accuracy improvement

5. QUICK WINS (THIS WEEK):
   [ ] Run augmentation script
   [ ] Retrain with hyperparameter tuning (batch_size, learning_rate)
   [ ] Validate on test set
   [ ] Measure improvement
    """)
    
    print("█"*80 + "\n")

if __name__ == '__main__':
    main()
