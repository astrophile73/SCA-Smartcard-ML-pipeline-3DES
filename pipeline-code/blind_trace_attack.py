"""
Main Script: Blind Trace Confidence Scoring (Post-Processing)

This script demonstrates how to:
1. Load predictions from blind trace inference
2. Aggregate predictions by card (Track2)
3. Score confidence based on multi-trace consistency
4. Flag uncertain cases for manual review
5. Output validated high-confidence keys

DESIGN: POST-PROCESSING (works on pre-computed predictions, not inference)
Use this AFTER inference pipeline has generated predictions.

Usage:
    python blind_trace_attack.py \
        --predictions_csv <path_to_predictions.csv> \
        --output_dir <output_path>
"""

import os
import sys
import argparse
import logging
from typing import Optional
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("blind_trace_attack")

# Add pipeline code to path
PIPELINE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_CODE_DIR)

from src.blind_trace_aggregation import aggregate_card_predictions, print_summary, save_aggregated_report


def run_blind_trace_attack(
    predictions_csv: str,
    output_dir: str,
    confidence_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Aggregate predictions from blind trace attack and compute confidence metrics.
    
    This is a POST-PROCESSING function that works after inference.
    
    Args:
        predictions_csv: CSV with inference results (3DES_KENC, 3DES_KMAC, 3DES_KDEK, Track2)
        output_dir: Output directory for results
        confidence_threshold: Agreement % for HIGH confidence
        
    Returns:
        DataFrame with aggregated predictions and confidence metrics
    """
    
    logger.info("="*100)
    logger.info("BLIND TRACE CONFIDENCE SCORING")
    logger.info("="*100)
    
    # Load predictions from inference
    logger.info(f"\n[1/3] Loading predictions from {predictions_csv}")
    predictions_df = pd.read_csv(predictions_csv)
    logger.info(f"  Loaded {len(predictions_df)} trace predictions")
    
    # Verify required columns
    required_cols = ['3DES_KENC', '3DES_KMAC', '3DES_KDEK']
    # Track2 or card_id
    card_col = 'Track2' if 'Track2' in predictions_df.columns else 'card_id'
    if card_col not in predictions_df.columns:
        raise ValueError(f"Predictions must have Track2 or card_id column")
    
    logger.info(f"  Grouping by: {card_col}")
    
    # Aggregate predictions by card
    logger.info(f"\n[2/3] Aggregating predictions by {card_col}")
    aggregated_df = aggregate_card_predictions(
        predictions_df,
        groupby_column=card_col,
        confidence_threshold=confidence_threshold
    )
    logger.info(f"  Aggregated into {len(aggregated_df)} unique cards")
    
    # Save aggregated results
    logger.info(f"\n[3/3] Saving results")
    os.makedirs(output_dir, exist_ok=True)
    save_aggregated_report(aggregated_df, output_dir)
    
    # Print summary
    print_summary(aggregated_df)
    
    return aggregated_df


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Blind Trace Confidence Scoring (post-processing)"
    )
    parser.add_argument('--predictions_csv', required=True,
                       help='CSV with inference predictions (3DES_KENC, 3DES_KMAC, 3DES_KDEK, Track2)')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for aggregated results')
    parser.add_argument('--confidence_threshold', type=float, default=0.8,
                       help='Agreement threshold for HIGH confidence (0-1)')
    
    args = parser.parse_args()
    
    # Run aggregation
    aggregated_df = run_blind_trace_attack(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold
    )
    
    logger.info("\n[SUCCESS] Confidence scoring completed!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
