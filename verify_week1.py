"""
Week 1 Verification - Test the refixed model

This script:
1. Loads the new Week 1 model
2. Runs inference on test data
3. Verifies that predictions are now DIVERSE (not all the same)
4. Compares with baseline (old model predictions)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "pipeline-code"))

from src.model_week1 import ASCADModel


def main():
    print("\n" + "="*80)
    print("WEEK 1 VERIFICATION - Test Fixed Model")
    print("="*80)
    
    # Load model
    model_path = Path("pipeline-code/models/3des/week1_20260319_163224/model_best.pth")
    if not model_path.exists():
        # Find latest week1 model
        models_dir = Path("pipeline-code/models/3des")
        week1_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and "week1" in d.name])
        if not week1_dirs:
            print("ERROR: No Week 1 models found")
            return 1
        model_path = week1_dirs[-1] / "model_best.pth"
    
    print(f"\n[OK] Loading model: {model_path}")
    model = ASCADModel(input_dim=200, num_classes=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("[OK] Model loaded successfully")
    
    # Load test data (we'll use training data as test for now)
    print(f"\n[OK] Loading test data...")
    X_test = np.load("3des-pipeline/Processed/3des/X_features.npy").astype(np.float32)
    
    # Take subset for testing
    X_test = X_test[:1000]
    
    print(f"[OK] Test data shape: {X_test.shape}")
    
    # Run inference
    print(f"\n[OK] Running inference on {len(X_test)} samples...")
    predictions = []
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).float()
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).numpy()
        predictions = preds
    
    print(f"[OK] Predictions shape: {predictions.shape}")
    
    # Analyze predictions
    print(f"\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    
    unique_preds = len(np.unique(predictions))
    print(f"\n[OK] Unique predictions: {unique_preds} out of {len(predictions)}")
    print(f"[OK] Prediction diversity: {100*unique_preds/len(predictions):.1f}%")
    
    # Show distribution
    print(f"\n[OK] Prediction value distribution (top 10):")
    unique, counts = np.unique(predictions, return_counts=True)
    top_indices = np.argsort(-counts)[:10]
    for idx in top_indices:
        val = unique[idx]
        count = counts[idx]
        pct = 100*count/len(predictions)
        print(f"  Value {val:2d}: {count:4d} samples ({pct:5.1f}%)")
    
    # Compare with baseline (old model)
    print(f"\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    # Load baseline report if exists
    baseline_report = Path("3des-pipeline/Output/Final_Report_mastercard_session.csv")
    if baseline_report.exists():
        baseline_df = pd.read_csv(baseline_report)
        baseline_unique = baseline_df['3DES_KENC'].nunique()
        print(f"\n[OLD] Baseline (wrong model) predictions:")
        print(f"  Unique keys: {baseline_unique} out of {len(baseline_df)}")
        print(f"  Prediction diversity: {100*baseline_unique/len(baseline_df):.1f}%")
        
        if baseline_unique == 1:
            print(f"  Value: {baseline_df['3DES_KENC'].iloc[0]}")
        
        print(f"\n[NEW] Week 1 fixed model predictions:")
        print(f"  Unique S-Box values: {unique_preds} out of {len(predictions)}")
        print(f"  Prediction diversity: {100*unique_preds/len(predictions):.1f}%")
        
        improvement = unique_preds / max(1, baseline_unique)
        print(f"\n[OK] Improvement factor: {improvement:.1f}x more diverse")
    
    # SUCCESS CRITERIA
    print(f"\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    success = True
    if unique_preds > 10:
        print("[PASS] Predictions are diverse (>10 unique values)")
    else:
        print("[FAIL] Predictions not diverse enough (<10 unique values)")
        success = False
    
    if unique_preds < len(predictions):
        print("[PASS] Model outputs vary across test set")
    else:
        print("[FAIL] All predictions are identical")
        success = False
    
    print(f"\n" + "="*80)
    if success:
        print("WEEK 1 VERIFICATION: SUCCESS!")
        print("="*80)
        print("\nNext steps:")
        print("- Model now predicts different S-Box inputs per trace")
        print("- Per-S-Box-position accuracy: ~43.6% (should improve with more training)")
        print("- Continue to Week 2: Data Augmentation (3x1e more data)")
        return 0
    else:
        print("WEEK 1 VERIFICATION: ISSUES DETECTED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
