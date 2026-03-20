"""
Mastercard 3DES Accuracy Verification Suite

Tests blind trace attack on Mastercard traces:
1. Load Mastercard 3DES traces (ignore RSA)
2. Run inference pipeline
3. Compare predictions with ground truth
4. Compute accuracy metrics
5. Verify 100% accuracy or report discrepancies
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
logger = logging.getLogger("mastercard_accuracy_tests")

# Add pipeline code to path
PIPELINE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_CODE_DIR)


def get_ground_truth_keys(input_dir: str, card_type: str = 'mastercard') -> dict:
    """
    Extract ground truth keys from CSV metadata.
    
    Expected columns in CSV:
    - T_DES_KENC, T_DES_KMAC, T_DES_KDEK (ground truth keys)
    - Track2 (card identifier)
    """
    logger.info(f"\n[TEST 1/6] Extracting ground truth keys...")
    
    ground_truth = {}
    csv_files = list(Path(input_dir).glob('traces_data_*[0-9]T_*.csv'))
    csv_files = [f for f in csv_files if not str(f).endswith('_rsa_')]
    csv_files = sorted(csv_files)[:5]  # Limit to first 5 files
    
    logger.info(f"  Found {len(csv_files)} 3DES CSV files")
    
    total_rows = 0
    unique_cards = set()
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, nrows=100)  # Sample rows
            logger.info(f"    Processing: {csv_file.name}")
            
            if 'T_DES_KENC' in df.columns:
                for idx, row in df.iterrows():
                    try:
                        track2 = str(row.get('Track2', f'unknown_{idx}'))
                        kenc = str(row['T_DES_KENC']).strip().upper()
                        kmac = str(row['T_DES_KMAC']).strip().upper()
                        kdek = str(row['T_DES_KDEK']).strip().upper()
                        
                        # Validate hex format
                        if len(kenc) == 16 and all(c in '0123456789ABCDEF' for c in kenc):
                            if track2 not in ground_truth:
                                ground_truth[track2] = {
                                    'KENC': kenc,
                                    'KMAC': kmac,
                                    'KDEK': kdek,
                                    'source_file': csv_file.name
                                }
                                unique_cards.add(track2)
                                total_rows += 1
                    except Exception as e:
                        pass
        except Exception as e:
            logger.warning(f"    Skipped {csv_file.name}: {str(e)[:100]}")
    
    logger.info(f"  ✓ Extracted ground truth for {len(ground_truth)} cards")
    logger.info(f"    Total traces sampled: {total_rows}")
    
    return ground_truth


def run_3des_inference(input_dir: str, model_dir: str, processed_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Run 3DES attack on Mastercard traces.
    """
    logger.info(f"\n[TEST 2/6] Running 3DES inference on Mastercard traces...")
    
    # Import pipeline modules
    from src.pipeline_3des import attack_3des
    from src.preprocess import preprocess_3des
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess 3DES traces
    logger.info(f"  Preprocessing 3DES traces...")
    try:
        processed_path, meta_path = preprocess_3des(
            input_dir,
            processed_dir,
            opt_dir=None,
            file_pattern='traces_data_*[0-9]T_*.csv',  # Only 3DES, no RSA
            card_type='mastercard'
        )
        logger.info(f"  ✓ Preprocessing complete")
    except Exception as e:
        logger.error(f"  Preprocessing failed: {str(e)[:200]}")
        return None
    
    # Run attack
    logger.info(f"  Running ML-CPA attack...")
    try:
        predicted_3des, final_key = attack_3des(
            processed_dir,
            model_dir,
            card_type='mastercard',
            target_key='session',
            return_confidence=True,
            n_attack=0,  # Use all traces
            pure_science=True
        )
        logger.info(f"  ✓ Attack complete")
        logger.info(f"    Predicted keys: {list(predicted_3des.keys()) if predicted_3des else 'None'}")
        
        return predicted_3des
    except Exception as e:
        logger.error(f"  Attack failed: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None


def extract_card_predictions(predicted_3des: dict, results_csv: str = None) -> dict:
    """
    Extract per-card predictions from inference results.
    
    Returns dict: {card_id: {'KENC': x, 'KMAC': y, 'KDEK': z}, ...}
    """
    logger.info(f"\n[TEST 3/6] Extracting card predictions...")
    
    predictions = {}
    
    if isinstance(predicted_3des, dict):
        # Direct dict result from attack
        for key_type, value in predicted_3des.items():
            if isinstance(value, dict):
                # Per-card predictions
                for card_id, key_val in value.items():
                    if card_id not in predictions:
                        predictions[card_id] = {}
                    predictions[card_id][key_type.replace('3DES_', '')] = str(key_val).upper()
            elif isinstance(value, str):
                # Single global prediction
                if 'global' not in predictions:
                    predictions['global'] = {}
                predictions['global'][key_type.replace('3DES_', '')] = str(value).upper()
    
    logger.info(f"  ✓ Extracted predictions for {len(predictions)} cards/groups")
    for card_id in list(predictions.keys())[:3]:
        logger.info(f"    {card_id}: {predictions[card_id]}")
    
    return predictions


def compute_accuracy(ground_truth: dict, predictions: dict) -> dict:
    """
    Compare predictions with ground truth.
    
    Returns: {
        'total_cards': int,
        'correct_kenc': int,
        'correct_kmac': int,
        'correct_kdek': int,
        'accuracy_kenc': float,
        'accuracy_kmac': float,
        'accuracy_kdek': float,
        'overall_accuracy': float,
        'discrepancies': list
    }
    """
    logger.info(f"\n[TEST 4/6] Computing accuracy metrics...")
    
    total = len(ground_truth)
    correct_kenc = 0
    correct_kmac = 0
    correct_kdek = 0
    discrepancies = []
    
    # Match predictions to ground truth
    for card_id, gt in ground_truth.items():
        # Try exact match first
        pred = predictions.get(card_id)
        
        # If exact match fails, try first prediction available
        if not pred and predictions:
            pred = next(iter(predictions.values()))
        
        if not pred:
            discrepancies.append({
                'card_id': card_id,
                'issue': 'No prediction found',
                'gt_kenc': gt['KENC'],
                'pred_kenc': 'N/A'
            })
            continue
        
        # Check each key type
        gt_kenc = gt['KENC'].upper()
        pred_kenc = pred.get('KENC', '').upper()
        
        if pred_kenc == gt_kenc:
            correct_kenc += 1
        else:
            discrepancies.append({
                'card_id': card_id,
                'key_type': 'KENC',
                'gt': gt_kenc,
                'pred': pred_kenc,
                'match': False
            })
        
        gt_kmac = gt['KMAC'].upper()
        pred_kmac = pred.get('KMAC', '').upper()
        if pred_kmac == gt_kmac:
            correct_kmac += 1
        else:
            discrepancies.append({
                'card_id': card_id,
                'key_type': 'KMAC',
                'gt': gt_kmac,
                'pred': pred_kmac,
                'match': False
            })
        
        gt_kdek = gt['KDEK'].upper()
        pred_kdek = pred.get('KDEK', '').upper()
        if pred_kdek == gt_kdek:
            correct_kdek += 1
        else:
            discrepancies.append({
                'card_id': card_id,
                'key_type': 'KDEK',
                'gt': gt_kdek,
                'pred': pred_kdek,
                'match': False
            })
    
    accuracy_kenc = 100.0 * correct_kenc / total if total > 0 else 0
    accuracy_kmac = 100.0 * correct_kmac / total if total > 0 else 0
    accuracy_kdek = 100.0 * correct_kdek / total if total > 0 else 0
    overall_accuracy = 100.0 * (correct_kenc + correct_kmac + correct_kdek) / (3 * total) if total > 0 else 0
    
    logger.info(f"  Total ground truth cards: {total}")
    logger.info(f"  KENC accuracy: {correct_kenc}/{total} ({accuracy_kenc:.1f}%)")
    logger.info(f"  KMAC accuracy: {correct_kmac}/{total} ({accuracy_kmac:.1f}%)")
    logger.info(f"  KDEK accuracy: {correct_kdek}/{total} ({accuracy_kdek:.1f}%)")
    logger.info(f"  Overall accuracy: {overall_accuracy:.1f}%")
    
    return {
        'total_cards': total,
        'correct_kenc': correct_kenc,
        'correct_kmac': correct_kmac,
        'correct_kdek': correct_kdek,
        'accuracy_kenc': accuracy_kenc,
        'accuracy_kmac': accuracy_kmac,
        'accuracy_kdek': accuracy_kdek,
        'overall_accuracy': overall_accuracy,
        'discrepancies': discrepancies
    }


def print_accuracy_report(accuracy_metrics: dict):
    """Print detailed accuracy report."""
    logger.info(f"\n[TEST 5/6] Accuracy Report")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("MASTERCARD 3DES ACCURACY VERIFICATION")
    print("="*80)
    
    print(f"\nTotal Cards Tested: {accuracy_metrics['total_cards']}")
    print(f"\nKey Accuracy by Type:")
    print(f"  KENC: {accuracy_metrics['correct_kenc']}/{accuracy_metrics['total_cards']} " 
          f"({accuracy_metrics['accuracy_kenc']:.1f}%)")
    print(f"  KMAC: {accuracy_metrics['correct_kmac']}/{accuracy_metrics['total_cards']} "
          f"({accuracy_metrics['accuracy_kmac']:.1f}%)")
    print(f"  KDEK: {accuracy_metrics['correct_kdek']}/{accuracy_metrics['total_cards']} "
          f"({accuracy_metrics['accuracy_kdek']:.1f}%)")
    
    print(f"\nOVERALL ACCURACY: {accuracy_metrics['overall_accuracy']:.1f}%")
    
    # Highlight whether 100% accuracy achieved
    if accuracy_metrics['overall_accuracy'] == 100.0:
        print("\n✅ 100% ACCURACY ACHIEVED!")
    elif accuracy_metrics['overall_accuracy'] >= 99.0:
        print(f"\n⚠️  NEAR-PERFECT ACCURACY: {accuracy_metrics['overall_accuracy']:.1f}%")
    else:
        print(f"\n❌ ACCURACY BELOW 100%: {accuracy_metrics['overall_accuracy']:.1f}%")
    
    # Report discrepancies
    if accuracy_metrics['discrepancies']:
        print(f"\nDiscrepancies Found: {len(accuracy_metrics['discrepancies'])}")
        print("-" * 80)
        for i, disc in enumerate(accuracy_metrics['discrepancies'][:10]):  # First 10
            if 'key_type' in disc:
                print(f"  Card {disc['card_id']} - {disc['key_type']}:")
                print(f"    Ground Truth: {disc['gt']}")
                print(f"    Predicted:   {disc['pred']}")
            else:
                print(f"  Card {disc['card_id']}: {disc['issue']}")
        if len(accuracy_metrics['discrepancies']) > 10:
            print(f"  ... and {len(accuracy_metrics['discrepancies']) - 10} more")
    else:
        print(f"\n✓ No discrepancies found")
    
    print("="*80)


def save_accuracy_report(accuracy_metrics: dict, output_dir: str):
    """Save accuracy report to CSV."""
    logger.info(f"\n[TEST 6/6] Saving accuracy report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary report
    summary = {
        'metric': [
            'Total Cards',
            'KENC Accuracy',
            'KMAC Accuracy',
            'KDEK Accuracy',
            'Overall Accuracy'
        ],
        'value': [
            f"{accuracy_metrics['total_cards']}",
            f"{accuracy_metrics['accuracy_kenc']:.1f}%",
            f"{accuracy_metrics['accuracy_kmac']:.1f}%",
            f"{accuracy_metrics['accuracy_kdek']:.1f}%",
            f"{accuracy_metrics['overall_accuracy']:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, 'mastercard_accuracy_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  ✓ Summary saved: {summary_path}")
    
    # Discrepancies
    if accuracy_metrics['discrepancies']:
        disc_df = pd.DataFrame(accuracy_metrics['discrepancies'])
        disc_path = os.path.join(output_dir, 'mastercard_discrepancies.csv')
        disc_df.to_csv(disc_path, index=False)
        logger.info(f"  ✓ Discrepancies saved: {disc_path}")


def main():
    """Run complete Mastercard accuracy verification."""
    logger.info("="*80)
    logger.info("MASTERCARD 3DES ACCURACY VERIFICATION SUITE")
    logger.info("="*80)
    
    # Directories
    input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
    model_dir = r"I:\freelance\SCA Smartcard ML Pipeline-3des\pipeline-code\models"
    processed_dir = os.path.join(PIPELINE_CODE_DIR, '..', 'Output', 'mastercard_processed')
    output_dir = os.path.join(PIPELINE_CODE_DIR, '..', 'Output', 'mastercard_accuracy')
    
    logger.info(f"\nInput directory: {input_dir}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Verify input exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)
    
    try:
        # Extract ground truth
        ground_truth = get_ground_truth_keys(input_dir, card_type='mastercard')
        if not ground_truth:
            logger.error("No ground truth keys extracted!")
            sys.exit(1)
        
        # Run inference
        predicted_3des = run_3des_inference(input_dir, model_dir, processed_dir, output_dir)
        if predicted_3des is None:
            logger.error("Inference failed!")
            sys.exit(1)
        
        # Extract predictions
        predictions = extract_card_predictions(predicted_3des)
        if not predictions:
            logger.error("No predictions extracted!")
            sys.exit(1)
        
        # Compute accuracy
        accuracy_metrics = compute_accuracy(ground_truth, predictions)
        
        # Print report
        print_accuracy_report(accuracy_metrics)
        
        # Save report
        save_accuracy_report(accuracy_metrics, output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("[SUCCESS] Mastercard accuracy verification completed!")
        logger.info("="*80)
        
        # Exit with status based on accuracy
        if accuracy_metrics['overall_accuracy'] == 100.0:
            logger.info("✅ 100% ACCURACY ACHIEVED!")
            sys.exit(0)
        else:
            logger.warning(f"⚠️  Accuracy: {accuracy_metrics['overall_accuracy']:.1f}%")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n[FAILED] Test failed with error:")
        logger.error(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
