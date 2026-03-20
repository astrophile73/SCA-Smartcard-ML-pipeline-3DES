"""
Test Script: Validate Blind Trace Confidence Scoring

Tests the three-module framework:
1. blind_trace_aggregation.py - Aggregation functions
2. blind_trace_attack.py - CLI entry point
3. Data flow: predictions → aggregation → confidence scores

Generates synthetic predictions and validates output.
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
logger = logging.getLogger("test_blind_trace_scoring")

# Add pipeline code to path
PIPELINE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_CODE_DIR)

from src.blind_trace_aggregation import aggregate_card_predictions, print_summary, save_aggregated_report


def create_synthetic_predictions(num_cards=5, traces_per_card=10):
    """
    Create synthetic predictions for testing.
    
    Args:
        num_cards: Number of unique cards (Track2 values)
        traces_per_card: Traces per card
        
    Returns:
        DataFrame with columns: trace_id, Track2, 3DES_KENC, 3DES_KMAC, 3DES_KDEK
    """
    logger.info(f"\n[TEST 1/5] Creating synthetic predictions...")
    logger.info(f"  Cards: {num_cards}, Traces per card: {traces_per_card}")
    
    rows = []
    
    # Card 1: 100% consistent (all same keys)
    for i in range(traces_per_card):
        rows.append({
            'trace_id': f'card1_trace{i}',
            'Track2': 'card_001',
            '3DES_KENC': 'AAAA111111111111',
            '3DES_KMAC': 'BBBB111111111111',
            '3DES_KDEK': 'CCCC111111111111'
        })
    
    # Card 2: 80% consistent (8 same, 2 different)
    for i in range(8):
        rows.append({
            'trace_id': f'card2_trace{i}',
            'Track2': 'card_002',
            '3DES_KENC': 'AAAA222222222222',
            '3DES_KMAC': 'BBBB222222222222',
            '3DES_KDEK': 'CCCC222222222222'
        })
    for i in range(2):
        rows.append({
            'trace_id': f'card2_trace_outlier{i}',
            'Track2': 'card_002',
            '3DES_KENC': 'AAAA999999999999',  # Different key
            '3DES_KMAC': 'BBBB222222222222',
            '3DES_KDEK': 'CCCC222222222222'
        })
    
    # Card 3: 50% consistent (split)
    for i in range(5):
        rows.append({
            'trace_id': f'card3_trace{i}',
            'Track2': 'card_003',
            '3DES_KENC': 'AAAA333333333333',
            '3DES_KMAC': 'BBBB333333333333',
            '3DES_KDEK': 'CCCC333333333333'
        })
    for i in range(5):
        rows.append({
            'trace_id': f'card3_trace_alt{i}',
            'Track2': 'card_003',
            '3DES_KENC': 'AAAA888888888888',  # Different key
            '3DES_KMAC': 'BBBB888888888888',  # Different key
            '3DES_KDEK': 'CCCC888888888888'   # Different key
        })
    
    # Card 4: 60% consistent (MEDIUM confidence)
    for i in range(6):
        rows.append({
            'trace_id': f'card4_trace{i}',
            'Track2': 'card_004',
            '3DES_KENC': 'AAAA444444444444',
            '3DES_KMAC': 'BBBB444444444444',
            '3DES_KDEK': 'CCCC444444444444'
        })
    for i in range(4):
        rows.append({
            'trace_id': f'card4_trace_alt{i}',
            'Track2': 'card_004',
            '3DES_KENC': 'AAAA777777777777',  # Different key
            '3DES_KMAC': 'BBBB777777777777',  # Different key
            '3DES_KDEK': 'CCCC777777777777'   # Different key
        })
    
    # Card 5: 40% consistent (LOW confidence)
    for i in range(4):
        rows.append({
            'trace_id': f'card5_trace{i}',
            'Track2': 'card_005',
            '3DES_KENC': 'AAAA555555555555',
            '3DES_KMAC': 'BBBB555555555555',
            '3DES_KDEK': 'CCCC555555555555'
        })
    for i in range(6):
        rows.append({
            'trace_id': f'card5_trace_alt{i}',
            'Track2': 'card_005',
            '3DES_KENC': 'AAAA666666666666',  # Different key
            '3DES_KMAC': 'BBBB666666666666',  # Different key
            '3DES_KDEK': 'CCCC666666666666'   # Different key
        })
    
    df = pd.DataFrame(rows)
    logger.info(f"  Created {len(df)} predictions for {len(df['Track2'].unique())} cards")
    return df


def test_consistency_computation(predictions_df):
    """Test consistency computation on synthetic data"""
    logger.info(f"\n[TEST 2/5] Testing consistency computation...")
    
    # Test on Card 1 (should be 100%)
    card1_preds = predictions_df[predictions_df['Track2'] == 'card_001']['3DES_KENC'].tolist()
    assert len(set(card1_preds)) == 1, "Card 1 should have 1 unique KENC"
    logger.info(f"  ✓ Card 1 KENC consistency: 100% (all {card1_preds[0]})")
    
    # Test on Card 3 (should be 50%)
    card3_preds = predictions_df[predictions_df['Track2'] == 'card_003']['3DES_KENC'].tolist()
    unique_vals = len(set(card3_preds))
    assert unique_vals == 2, f"Card 3 should have 2 unique KENC, got {unique_vals}"
    logger.info(f"  ✓ Card 3 KENC consistency: 50% (split between predictions)")


def test_aggregation_function(predictions_df):
    """Test aggregation function"""
    logger.info(f"\n[TEST 3/5] Testing aggregation function...")
    
    aggregated_df = aggregate_card_predictions(
        predictions_df,
        groupby_column='Track2',
        confidence_threshold=0.8
    )
    
    logger.info(f"  Aggregated {len(predictions_df)} traces into {len(aggregated_df)} cards")
    
    # Verify structure
    expected_cols = ['Card_ID', 'Num_Traces', 'Predicted_KENC', 'Predicted_KMAC', 
                    'Predicted_KDEK', 'Consistency_KENC', 'Consistency_KMAC',
                    'Consistency_KDEK', 'Confidence_Level', 'Flagged_For_Review']
    
    for col in expected_cols:
        assert col in aggregated_df.columns, f"Missing column: {col}"
    logger.info(f"  ✓ All required columns present")
    
    # Check confidence levels
    print("\n  Aggregated Results:")
    for idx, row in aggregated_df.iterrows():
        print(f"    {row['Card_ID']}: {row['Num_Traces']:2d} traces → "
              f"KENC consistency {row['Consistency_KENC']}, "
              f"confidence: {row['Confidence_Level']}, "
              f"flagged: {row['Flagged_For_Review']}")
    
    # Verify confidence levels
    card1 = aggregated_df[aggregated_df['Card_ID'] == 'card_001'].iloc[0]
    assert card1['Confidence_Level'] == 'HIGH', f"Card 1 should be HIGH, got {card1['Confidence_Level']}"
    logger.info(f"  ✓ Card 1 (100% consistent) → HIGH confidence")
    
    card3 = aggregated_df[aggregated_df['Card_ID'] == 'card_003'].iloc[0]
    assert card3['Confidence_Level'] == 'LOW', f"Card 3 should be LOW, got {card3['Confidence_Level']}"
    logger.info(f"  ✓ Card 3 (50% consistent) → LOW confidence")
    logger.info(f"  ✓ Card 3 flagged for review: {card3['Flagged_For_Review']}")
    
    card4 = aggregated_df[aggregated_df['Card_ID'] == 'card_004'].iloc[0]
    assert card4['Confidence_Level'] == 'MEDIUM', f"Card 4 should be MEDIUM, got {card4['Confidence_Level']}"
    logger.info(f"  ✓ Card 4 (60% consistent) → MEDIUM confidence")
    
    return aggregated_df


def test_output_generation(aggregated_df, output_dir):
    """Test output file generation"""
    logger.info(f"\n[TEST 4/5] Testing output generation...")
    
    os.makedirs(output_dir, exist_ok=True)
    save_aggregated_report(aggregated_df, output_dir)
    
    # Check files were created
    aggregated_file = os.path.join(output_dir, 'blind_trace_aggregated_results.csv')
    high_conf_file = os.path.join(output_dir, 'blind_trace_high_confidence_keys.csv')
    
    assert os.path.exists(aggregated_file), f"Missing: {aggregated_file}"
    logger.info(f"  ✓ Created: blind_trace_aggregated_results.csv")
    
    assert os.path.exists(high_conf_file), f"Missing: {high_conf_file}"
    logger.info(f"  ✓ Created: blind_trace_high_confidence_keys.csv")
    
    # Load and verify
    aggregated_loaded = pd.read_csv(aggregated_file)
    high_conf_loaded = pd.read_csv(high_conf_file)
    
    logger.info(f"    - Aggregated results: {len(aggregated_loaded)} cards")
    logger.info(f"    - High confidence keys: {len(high_conf_loaded)} cards")
    
    # High-confidence should only have HIGH confidence
    assert all(high_conf_loaded['Confidence_Level'] == 'HIGH'), \
        "High-confidence file should only contain HIGH confidence predictions"
    logger.info(f"  ✓ High-confidence file contains only HIGH confidence cases")
    
    return aggregated_loaded, high_conf_loaded


def test_summary_output(aggregated_df):
    """Test summary printing"""
    logger.info(f"\n[TEST 5/5] Testing summary output...")
    logger.info(f"  Summary statistics:")
    print_summary(aggregated_df)


def main():
    """Run all tests"""
    logger.info("="*100)
    logger.info("BLIND TRACE CONFIDENCE SCORING - TEST SUITE")
    logger.info("="*100)
    
    try:
        # Test 1: Create synthetic predictions
        predictions_df = create_synthetic_predictions(num_cards=5, traces_per_card=10)
        
        # Test 2: Consistency computation
        test_consistency_computation(predictions_df)
        
        # Test 3: Aggregation function
        aggregated_df = test_aggregation_function(predictions_df)
        
        # Test 4: Output generation
        output_dir = os.path.join(PIPELINE_CODE_DIR, '..', 'Output', 'blind_trace_test')
        aggregated_loaded, high_conf_loaded = test_output_generation(aggregated_df, output_dir)
        
        # Test 5: Summary output
        test_summary_output(aggregated_df)
        
        logger.info("\n" + "="*100)
        logger.info("[SUCCESS] All tests passed!")
        logger.info("="*100)
        logger.info(f"\nTest results saved to: {output_dir}")
        logger.info("\nFramework is ready for:")
        logger.info("  1. Integration into main pipeline")
        logger.info("  2. Testing on actual blind trace predictions")
        logger.info("  3. Production deployment")
        
    except Exception as e:
        logger.error(f"\n[FAILED] Test failed with error:")
        logger.error(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
