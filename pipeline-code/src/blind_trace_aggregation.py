"""
Practical blind trace aggregation and confidence scoring

This module aggregates predictions from multiple traces of the same card
and computes consistency-based confidence metrics.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
import pandas as pd
import numpy as np

logger = logging.getLogger("blind_trace_aggregation")


def aggregate_card_predictions(predictions_df: pd.DataFrame,
                               groupby_column: str = 'Track2',
                               confidence_threshold: float = 0.8) -> pd.DataFrame:
    """
    Aggregate predictions from multiple traces of the same card.
    
    Args:
        predictions_df: DataFrame with predictions (must have 3DES_KENC, 3DES_KMAC, 3DES_KDEK, Track2/card_id)
        groupby_column: Column to group traces by (e.g., 'Track2', 'card_id')
        confidence_threshold: % agreement above which we assign HIGH confidence
        
    Returns:
        DataFrame with aggregated predictions and confidence metrics
    """
    aggregated = []
    
    for card_id, group_df in predictions_df.groupby(groupby_column):
        # Get predictions for this card
        kenc_preds = group_df['3DES_KENC'].tolist()
        kmac_preds = group_df['3DES_KMAC'].tolist()
        kdek_preds = group_df['3DES_KDEK'].tolist()
        
        # Compute consistency
        kenc_consistency, kenc_best = _compute_consistency(kenc_preds)
        kmac_consistency, kmac_best = _compute_consistency(kmac_preds)
        kdek_consistency, kdek_best = _compute_consistency(kdek_preds)
        
        # Determine confidence level
        avg_consistency = np.mean([kenc_consistency, kmac_consistency, kdek_consistency])
        
        if avg_consistency >= confidence_threshold:
            confidence = "HIGH"
            flagged = False
        elif avg_consistency >= 0.6:
            confidence = "MEDIUM"
            flagged = False
        else:
            confidence = "LOW"
            flagged = True
        
        aggregated.append({
            'Card_ID': card_id,
            'Num_Traces': len(group_df),
            'Predicted_KENC': kenc_best,
            'Consistency_KENC': f"{kenc_consistency:.1%}",
            'Agreement_KENC': f"{int(kenc_consistency * len(group_df))}/{len(group_df)}",
            'Predicted_KMAC': kmac_best,
            'Consistency_KMAC': f"{kmac_consistency:.1%}",
            'Agreement_KMAC': f"{int(kmac_consistency * len(group_df))}/{len(group_df)}",
            'Predicted_KDEK': kdek_best,
            'Consistency_KDEK': f"{kdek_consistency:.1%}",
            'Agreement_KDEK': f"{int(kdek_consistency * len(group_df))}/{len(group_df)}",
            'Avg_Consistency': f"{avg_consistency:.1%}",
            'Confidence_Level': confidence,
            'Flagged_For_Review': 'YES' if flagged else 'NO'
        })
    
    return pd.DataFrame(aggregated)


def _compute_consistency(predictions: List[str]) -> Tuple[float, str]:
    """
    Compute consistency for a list of predictions
    
    Returns:
        (consistency_ratio, most_common_prediction)
    """
    if not predictions:
        return 0.0, "unknown"
    
    counter = Counter(predictions)
    most_common, count = counter.most_common(1)[0]
    consistency = count / len(predictions)
    
    return consistency, most_common


def print_summary(aggregated_df: pd.DataFrame):
    """Print human-readable summary"""
    print("\n" + "="*100)
    print("BLIND TRACE AGGREGATION SUMMARY")
    print("="*100)
    
    total_cards = len(aggregated_df)
    high_conf = len(aggregated_df[aggregated_df['Confidence_Level'] == 'HIGH'])
    medium_conf = len(aggregated_df[aggregated_df['Confidence_Level'] == 'MEDIUM'])
    low_conf = len(aggregated_df[aggregated_df['Confidence_Level'] == 'LOW'])
    flagged = len(aggregated_df[aggregated_df['Flagged_For_Review'] == 'YES'])
    
    print(f"\nTotal Cards: {total_cards}")
    print(f"  HIGH confidence:   {high_conf} ({100*high_conf/total_cards:.1f}%)")
    print(f"  MEDIUM confidence: {medium_conf} ({100*medium_conf/total_cards:.1f}%)")
    print(f"  LOW confidence:    {low_conf} ({100*low_conf/total_cards:.1f}%)")
    print(f"  Flagged for review: {flagged} ({100*flagged/total_cards:.1f}%)")
    
    print("\n" + "-"*100)
    print("HIGH CONFIDENCE PREDICTIONS")
    print("-"*100)
    
    high_df = aggregated_df[aggregated_df['Confidence_Level'] == 'HIGH']
    if len(high_df) > 0:
        for _, row in high_df.iterrows():
            print(f"\nCard: {row['Card_ID']}")
            print(f"  Traces analyzed: {row['Num_Traces']}")
            print(f"  KENC: {row['Predicted_KENC']} ({row['Consistency_KENC']}, {row['Agreement_KENC']} agreement)")
            print(f"  KMAC: {row['Predicted_KMAC']} ({row['Consistency_KMAC']}, {row['Agreement_KMAC']} agreement)")
            print(f"  KDEK: {row['Predicted_KDEK']} ({row['Consistency_KDEK']}, {row['Agreement_KDEK']} agreement)")
    else:
        print("No high confidence predictions")
    
    if flagged > 0:
        print("\n" + "-"*100)
        print("FLAGGED FOR REVIEW (LOW CONFIDENCE)")
        print("-"*100)
        
        flagged_df = aggregated_df[aggregated_df['Flagged_For_Review'] == 'YES']
        for _, row in flagged_df.iterrows():
            print(f"\nCard: {row['Card_ID']}")
            print(f"  Traces analyzed: {row['Num_Traces']}")
            print(f"  Avg consistency: {row['Avg_Consistency']}")
            print(f"  KENC: {row['Predicted_KENC']} ({row['Consistency_KENC']})")
            print(f"  KMAC: {row['Predicted_KMAC']} ({row['Consistency_KMAC']})")
            print(f"  KDEK: {row['Predicted_KDEK']} ({row['Consistency_KDEK']})")
            print(f"  WARNING: Low consistency - results may be unreliable")
    
    print("\n" + "="*100)


def save_aggregated_report(aggregated_df: pd.DataFrame, output_dir: str):
    """Save aggregated results to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'blind_trace_aggregated_results.csv')
    aggregated_df.to_csv(output_path, index=False)
    logger.info(f"Aggregated results saved to {output_path}")
    
    # Also save high-confidence-only report
    high_conf_df = aggregated_df[aggregated_df['Confidence_Level'] == 'HIGH']
    if len(high_conf_df) > 0:
        high_conf_path = os.path.join(output_dir, 'blind_trace_high_confidence_keys.csv')
        high_conf_df.to_csv(high_conf_path, index=False)
        logger.info(f"High confidence keys saved to {high_conf_path}")


if __name__ == "__main__":
    """
    Example usage:
    
    # After running inference on blind traces and collecting predictions:
    predictions_df = pd.read_csv("blind_predictions.csv")
    
    aggregated = aggregate_card_predictions(
        predictions_df,
        groupby_column='Track2',
        confidence_threshold=0.8
    )
    
    print_summary(aggregated)
    save_aggregated_report(aggregated, "output_dir")
    """
    pass
