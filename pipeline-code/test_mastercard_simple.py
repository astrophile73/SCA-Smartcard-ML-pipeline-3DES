"""
Simplified Mastercard 3DES Accuracy Test

Runs 3DES attack on Mastercard traces and verifies 100% accuracy.
Compares predicted keys with ground truth keys from CSV metadata.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mastercard_test")

PIPELINE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_CODE_DIR)


def load_ground_truth(input_dir: str) -> dict:
    """Load ground truth keys from first CSV file."""
    logger.info("Loading ground truth keys...")
    
    csv_file = list(Path(input_dir).glob('traces_data_1000T_1.csv'))[0]
    df = pd.read_csv(csv_file, nrows=50)  # Load first 50 rows
    
    gt = {}
    for idx, row in df.iterrows():
        track2 = str(row['Track2']).strip()
        kenc = str(row['T_DES_KENC']).strip().upper()
        kmac = str(row['T_DES_KMAC']).strip().upper()
        kdek = str(row['T_DES_KDEK']).strip().upper()
        
        gt[track2] = {'KENC': kenc, 'KMAC': kmac, 'KDEK': kdek}
    
    logger.info(f"✓ Loaded {len(gt)} ground truth keys")
    print("Sample ground truth:")
    for card_id in list(gt.keys())[:3]:
        print(f"  {card_id}: {gt[card_id]['KENC']}")
    
    return gt


def run_attack(input_dir: str, model_dir: str, processed_dir: str, opt_dir: str) -> dict:
    """Run the 3DES attack pipeline."""
    logger.info("\nRunning 3DES attack...")
    
    try:
        from src.pipeline_3des import attack_3des, preprocess_3des
        
        # Preprocess
        logger.info("  Preprocessing 3DES traces...")
        os.makedirs(processed_dir, exist_ok=True)
        processed_path, meta_path = preprocess_3des(
            input_dir,
            processed_dir,
            opt_dir=opt_dir,
            file_pattern='traces_data_*[0-9]T_*.csv',  # Only 3DES
            card_type='mastercard',
            use_existing_pois=True,
            include_secrets=True
        )
        logger.info(f"  ✓ Preprocessed to {processed_dir}")
        
        # Attack
        logger.info("  Running ML-CPA attack...")
        predicted_3des, final_key = attack_3des(
            processed_dir,
            model_dir,
            card_type='mastercard',
            target_key='session',
            return_confidence=False,
            n_attack=0,
            pure_science=True
        )
        
        logger.info(f"  ✓ Attack complete")
        return predicted_3des, meta_path
        
    except Exception as e:
        logger.error(f"Attack failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_results(ground_truth: dict, predicted_3des: dict, meta_path: str) -> dict:
    """
    Compare predicted keys with ground truth.
    Load metadata to match predictions with ground truth by card.
    """
    logger.info("\nComparing predictions with ground truth...")
    
    # Load meta to get card IDs for each trace
    meta_df = pd.read_csv(meta_path)
    logger.info(f"  Loaded metadata with {len(meta_df)} traces")
    
    # Group by card to get average predictions
    card_predictions = {}
    
    for idx, row in meta_df.iterrows():
        track2 = str(row.get('Track2', row.get('card_id', 'unknown'))).strip()
        
        # Extract predicted keys from row
        pred_kenc = str(row.get('Predicted_3DES_KENC', row.get('3DES_KENC', ''))).strip().upper()
        pred_kmac = str(row.get('Predicted_3DES_KMAC', row.get('3DES_KMAC', ''))).strip().upper()
        pred_kdek = str(row.get('Predicted_3DES_KDEK', row.get('3DES_KDEK', ''))).strip().upper()
        
        if track2 not in card_predictions:
            card_predictions[track2] = []
        
        card_predictions[track2].append({
            'KENC': pred_kenc,
            'KMAC': pred_kmac,
            'KDEK': pred_kdek
        })
    
    # Compare
    results = {
        'total': 0,
        'correct_kenc': 0,
        'correct_kmac': 0,
        'correct_kdek': 0,
        'discrepancies': []
    }
    
    for card_id, gt in ground_truth.items():
        if card_id not in card_predictions:
            results['discrepancies'].append({'card': card_id, 'issue': 'No prediction found'})
            continue
        
        # Use first prediction (or most common?)
        preds = card_predictions[card_id]
        pred = preds[0]  # Use first trace's prediction
        
        results['total'] += 1
        
        # Check each key type
        if pred['KENC'] == gt['KENC']:
            results['correct_kenc'] += 1
        else:
            results['discrepancies'].append({
                'card': card_id,
                'key': 'KENC',
                'gt': gt['KENC'],
                'pred': pred['KENC']
            })
        
        if pred['KMAC'] == gt['KMAC']:
            results['correct_kmac'] += 1
        else:
            results['discrepancies'].append({
                'card': card_id,
                'key': 'KMAC',
                'gt': gt['KMAC'],
                'pred': pred['KMAC']
            })
        
        if pred['KDEK'] == gt['KDEK']:
            results['correct_kdek'] += 1
        else:
            results['discrepancies'].append({
                'card': card_id,
                'key': 'KDEK',
                'gt': gt['KDEK'],
                'pred': pred['KDEK']
            })
    
    return results


def print_results(results: dict):
    """Print detailed results."""
    total = results['total']
    
    print("\n" + "="*80)
    print("MASTERCARD 3DES ACCURACY VERIFICATION")
    print("="*80)
    print(f"\nCards tested: {total}")
    print(f"\nAccuracy by Key Type:")
    print(f"  KENC: {results['correct_kenc']}/{total} ({100*results['correct_kenc']/total:.1f}%)")
    print(f"  KMAC: {results['correct_kmac']}/{total} ({100*results['correct_kmac']/total:.1f}%)")
    print(f"  KDEK: {results['correct_kdek']}/{total} ({100*results['correct_kdek']/total:.1f}%)")
    
    overall = (results['correct_kenc'] + results['correct_kmac'] + results['correct_kdek']) / (3 * total)
    print(f"\nOVERALL ACCURACY: {overall*100:.1f}%")
    
    if overall == 1.0:
        print("✅ 100% ACCURACY ACHIEVED!")
    else:
        print(f"❌ Accuracy: {overall*100:.1f}%")
    
    if results['discrepancies']:
        print(f"\nDiscrepancies: {len(results['discrepancies'])}")
        for disc in results['discrepancies'][:10]:
            print(f"  {disc}")
    
    print("="*80)


def main():
    # Directories
    input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
    model_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\pipeline-code\models"
    processed_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\Output\mastercard_processed"
    opt_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\Optimization"
    
    logger.info("="*80)
    logger.info("MASTERCARD 3DES ACCURACY TEST")
    logger.info("="*80)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Models: {model_dir}")
    
    # Load ground truth
    ground_truth = load_ground_truth(input_dir)
    
    # Run attack
    predicted_3des, meta_path = run_attack(input_dir, model_dir, processed_dir, opt_dir)
    
    if predicted_3des is None or meta_path is None:
        logger.error("Attack failed!")
        sys.exit(1)
    
    # Compare
    results = compare_results(ground_truth, predicted_3des, meta_path)
    
    # Print
    print_results(results)
    
    # Save
    os.makedirs(r"I:\freelance\SCA Smartcard ML Pipeline-3des\Output\mastercard_test", exist_ok=True)
    summary_df = pd.DataFrame({
        'Metric': ['Total Cards', 'KENC Accuracy', 'KMAC Accuracy', 'KDEK Accuracy'],
        'Value': [
            f"{results['total']}",
            f"{results['correct_kenc']}/{results['total']}",
            f"{results['correct_kmac']}/{results['total']}",
            f"{results['correct_kdek']}/{results['total']}"
        ]
    })
    summary_df.to_csv(r"I:\freelance\SCA Smartcard ML Pipeline-3des\Output\mastercard_test\accuracy_summary.csv", index=False)
    logger.info(f"\n✓ Results saved to Output/mastercard_test/")


if __name__ == "__main__":
    main()
