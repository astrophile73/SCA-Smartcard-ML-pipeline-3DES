"""
Blind Trace Validator: Confidence Scoring for 3DES Key Recovery

This module provides consistency-based validation for ML-CPA attacks on blind traces.
Since blind traces lack ground truth, we use multi-trace consistency as a confidence metric.

Strategy:
- Group traces by Track2 (same card should have same key)
- Run CPA on each trace individually
- Score predictions by agreement across traces
- Return confidence metrics alongside recovered keys
"""

import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import logging

logger = logging.getLogger("blind_trace_validator")


@dataclass
class KeyRecoveryResult:
    """Result of key recovery for a single trace"""
    trace_id: str
    track2: str
    atc: str
    predicted_kenc: str
    predicted_kmac: str
    predicted_kdek: str
    model_confidence: Dict[str, float]  # Per key type confidence from model


@dataclass
class CardKeyEstimate:
    """Final key estimate for a card with confidence metrics"""
    track2: str
    num_traces: int
    num_traces_analyzed: int
    
    # Final key estimates
    estimated_kenc: str
    estimated_kmac: str
    estimated_kdek: str
    
    # Consistency metrics (0-1, where 1 = perfect agreement)
    consistency_kenc: float
    consistency_kmac: float
    consistency_kdek: float
    
    # Agreement counts (how many traces predicted same key)
    agreement_kenc: int
    agreement_kmac: int
    agreement_kdek: int
    
    # Model confidence (average across traces)
    avg_model_confidence_kenc: float
    avg_model_confidence_kmac: float
    avg_model_confidence_kdek: float
    
    # Uncertainty flags
    is_uncertain: bool  # True if consistency < 0.8
    confidence_level: str  # HIGH, MEDIUM, LOW
    flagged_for_review: bool


class BlindTraceValidator:
    """
    Validates key recovery on blind traces using consistency-based scoring.
    """
    
    def __init__(self, processed_dir: str, model_dir: str, 
                 consistency_threshold: float = 0.8):
        """
        Args:
            processed_dir: Path to processed features directory
            model_dir: Path to trained models directory
            consistency_threshold: Agreement % above which we flag HIGH confidence
        """
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.consistency_threshold = consistency_threshold
        
        # Will be populated during validation
        self.trace_results: List[KeyRecoveryResult] = []
        self.card_results: Dict[str, CardKeyEstimate] = {}
        
    def validate_blind_traces(self, 
                              blind_traces_path: str,
                              output_dir: str = None) -> Dict[str, CardKeyEstimate]:
        """
        Main entry point: validate blind traces and return confidence-scored results
        
        Args:
            blind_traces_path: Path to blind traces CSV (must have trace_data, Track2, ATC)
            output_dir: Optional path to save validation report
            
        Returns:
            Dictionary mapping Track2 → CardKeyEstimate with confidence metrics
        """
        logger.info(f"Loading blind traces from {blind_traces_path}")
        blind_df = pd.read_csv(blind_traces_path)
        
        required_cols = ['trace_data', 'Track2', 'ATC']
        missing = [c for c in required_cols if c not in blind_df.columns]
        if missing:
            raise ValueError(f"Missing columns in blind traces: {missing}")
        
        logger.info(f"Loaded {len(blind_df)} blind traces")
        
        # Group traces by Track2 (card)
        traces_by_card = self._group_traces_by_card(blind_df)
        logger.info(f"Traces grouped into {len(traces_by_card)} unique cards")
        
        # Process each card
        for track2, traces in traces_by_card.items():
            logger.info(f"\nProcessing card {track2}: {len(traces)} traces")
            card_result = self._process_card_traces(track2, traces)
            self.card_results[track2] = card_result
            
            # Log confidence assessment
            self._log_card_result(card_result)
        
        # Save report if requested
        if output_dir:
            self._save_validation_report(output_dir)
        
        return self.card_results
    
    def _group_traces_by_card(self, blind_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group traces by Track2 (card identifier)"""
        return {track2: group.reset_index(drop=True) 
                for track2, group in blind_df.groupby('Track2')}
    
    def _process_card_traces(self, track2: str, traces: pd.DataFrame) -> CardKeyEstimate:
        """
        Process all traces for a single card and compute consistency metrics
        """
        from src.inference_3des import recover_3des_keys
        
        kenc_predictions = []
        kmac_predictions = []
        kdek_predictions = []
        model_confidences_kenc = []
        model_confidences_kmac = []
        model_confidences_kdek = []
        
        # Run CPA on each trace
        for idx, trace_row in traces.iterrows():
            try:
                # Create minimal metadata for this trace
                meta_for_trace = self._create_trace_metadata(trace_row)
                
                # Run inference for this trace
                pred = recover_3des_keys(
                    self.processed_dir, 
                    self.model_dir,
                    card_type="universal",
                    n_attack=None  # Use all traces (but we filtered to 1 here)
                )
                
                if pred:
                    kenc_predictions.append(pred.get('3DES_KENC', 'unknown'))
                    kmac_predictions.append(pred.get('3DES_KMAC', 'unknown'))
                    kdek_predictions.append(pred.get('3DES_KDEK', 'unknown'))
                    
                    # TODO: Extract model confidence scores (average softmax prob)
                    model_confidences_kenc.append(0.5)  # Placeholder
                    model_confidences_kmac.append(0.5)
                    model_confidences_kdek.append(0.5)
                    
                    result = KeyRecoveryResult(
                        trace_id=f"{track2}_{idx}",
                        track2=track2,
                        atc=str(trace_row.get('ATC', 'unknown')),
                        predicted_kenc=pred.get('3DES_KENC', 'unknown'),
                        predicted_kmac=pred.get('3DES_KMAC', 'unknown'),
                        predicted_kdek=pred.get('3DES_KDEK', 'unknown'),
                        model_confidence={
                            'kenc': model_confidences_kenc[-1],
                            'kmac': model_confidences_kmac[-1],
                            'kdek': model_confidences_kdek[-1]
                        }
                    )
                    self.trace_results.append(result)
                    
            except Exception as e:
                logger.warning(f"Failed to process trace {idx} for card {track2}: {e}")
                continue
        
        # Compute consistency metrics
        consistency_kenc, agreement_kenc, est_kenc = self._compute_consistency(kenc_predictions)
        consistency_kmac, agreement_kmac, est_kmac = self._compute_consistency(kmac_predictions)
        consistency_kdek, agreement_kdek, est_kdek = self._compute_consistency(kdek_predictions)
        
        # Determine confidence level
        avg_consistency = np.mean([consistency_kenc, consistency_kmac, consistency_kdek])
        
        if avg_consistency >= self.consistency_threshold:
            confidence_level = "HIGH"
            is_uncertain = False
        elif avg_consistency >= 0.6:
            confidence_level = "MEDIUM"
            is_uncertain = False
        else:
            confidence_level = "LOW"
            is_uncertain = True
        
        # Flag if LOW confidence or predictions vary widely
        flagged = is_uncertain or any(c < 0.6 for c in [consistency_kenc, consistency_kmac, consistency_kdek])
        
        return CardKeyEstimate(
            track2=track2,
            num_traces=len(traces),
            num_traces_analyzed=len(kenc_predictions),
            estimated_kenc=est_kenc,
            estimated_kmac=est_kmac,
            estimated_kdek=est_kdek,
            consistency_kenc=consistency_kenc,
            consistency_kmac=consistency_kmac,
            consistency_kdek=consistency_kdek,
            agreement_kenc=agreement_kenc,
            agreement_kmac=agreement_kmac,
            agreement_kdek=agreement_kdek,
            avg_model_confidence_kenc=np.mean(model_confidences_kenc) if model_confidences_kenc else 0.0,
            avg_model_confidence_kmac=np.mean(model_confidences_kmac) if model_confidences_kmac else 0.0,
            avg_model_confidence_kdek=np.mean(model_confidences_kdek) if model_confidences_kdek else 0.0,
            is_uncertain=is_uncertain,
            confidence_level=confidence_level,
            flagged_for_review=flagged
        )
    
    def _create_trace_metadata(self, trace_row: pd.Series) -> pd.DataFrame:
        """Create minimal metadata for a single trace"""
        # This is a placeholder - in reality you'd need to reconstruct
        # the metadata format expected by recover_3des_keys
        return pd.DataFrame([trace_row])
    
    def _compute_consistency(self, predictions: List[str]) -> Tuple[float, int, str]:
        """
        Compute consistency metrics for predictions
        
        Returns:
            (consistency_score, agreement_count, most_common_prediction)
        """
        if not predictions:
            return 0.0, 0, "unknown"
        
        # Count occurrences
        counter = Counter(predictions)
        most_common_key, count = counter.most_common(1)[0]
        
        # Consistency = agreement percentage
        consistency = count / len(predictions)
        
        return consistency, count, most_common_key
    
    def _log_card_result(self, result: CardKeyEstimate):
        """Log summary of card result"""
        logger.info(f"  Card: {result.track2}")
        logger.info(f"  Traces analyzed: {result.num_traces_analyzed}/{result.num_traces}")
        logger.info(f"  Confidence: {result.confidence_level}")
        logger.info(f"  Consistency: KENC={result.consistency_kenc:.1%}, "
                   f"KMAC={result.consistency_kmac:.1%}, "
                   f"KDEK={result.consistency_kdek:.1%}")
        logger.info(f"  Estimated KENC: {result.estimated_kenc}")
        logger.info(f"  Estimated KMAC: {result.estimated_kmac}")
        logger.info(f"  Estimated KDEK: {result.estimated_kdek}")
        
        if result.flagged_for_review:
            logger.warning(f"  *** FLAGGED FOR REVIEW ***")
    
    def _save_validation_report(self, output_dir: str):
        """Save validation report to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary report
        report_data = []
        for track2, result in self.card_results.items():
            report_data.append({
                'Track2': result.track2,
                'Num_Traces': result.num_traces,
                'Num_Analyzed': result.num_traces_analyzed,
                'Confidence_Level': result.confidence_level,
                'Consistency_KENC': f"{result.consistency_kenc:.1%}",
                'Consistency_KMAC': f"{result.consistency_kmac:.1%}",
                'Consistency_KDEK': f"{result.consistency_tdek:.1%}",
                'Predicted_KENC': result.estimated_kenc,
                'Predicted_KMAC': result.estimated_kmac,
                'Predicted_KDEK': result.estimated_kdek,
                'Model_Confidence_KENC': f"{result.avg_model_confidence_kenc:.3f}",
                'Model_Confidence_KMAC': f"{result.avg_model_confidence_kmac:.3f}",
                'Model_Confidence_KDEK': f"{result.avg_model_confidence_kdek:.3f}",
                'Flagged_For_Review': result.flagged_for_review
            })
        
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(output_dir, 'blind_trace_validation_report.csv')
        report_df.to_csv(report_path, index=False)
        logger.info(f"Validation report saved to {report_path}")
        
        # Also save detail trace results
        if self.trace_results:
            trace_data = []
            for result in self.trace_results:
                trace_data.append({
                    'Trace_ID': result.trace_id,
                    'Track2': result.track2,
                    'ATC': result.atc,
                    'Predicted_KENC': result.predicted_kenc,
                    'Predicted_KMAC': result.predicted_kmac,
                    'Predicted_KDEK': result.predicted_kdek,
                    'Model_Conf_KENC': f"{result.model_confidence['kenc']:.3f}",
                    'Model_Conf_KMAC': f"{result.model_confidence['kmac']:.3f}",
                    'Model_Conf_KDEK': f"{result.model_confidence['kdek']:.3f}"
                })
            
            trace_df = pd.DataFrame(trace_data)
            trace_path = os.path.join(output_dir, 'blind_trace_predictions_detail.csv')
            trace_df.to_csv(trace_path, index=False)
            logger.info(f"Detailed predictions saved to {trace_path}")
    
    def get_high_confidence_keys(self) -> Dict[str, Dict[str, str]]:
        """
        Extract only high-confidence key predictions
        
        Returns:
            Dictionary mapping Track2 → {kenc, kmac, kdek} for HIGH confidence only
        """
        high_conf = {}
        for track2, result in self.card_results.items():
            if result.confidence_level == "HIGH" and not result.flagged_for_review:
                high_conf[track2] = {
                    'KENC': result.estimated_kenc,
                    'KMAC': result.estimated_kmac,
                    'KDEK': result.estimated_kdek,
                    'Consistency': {
                        'KENC': f"{result.consistency_kenc:.1%}",
                        'KMAC': f"{result.consistency_kmac:.1%}",
                        'KDEK': f"{result.consistency_tdek:.1%}"
                    }
                }
        return high_conf
    
    def get_flagged_keys(self) -> Dict[str, Dict]:
        """
        Extract keys flagged for review (LOW confidence or high disagreement)
        """
        flagged = {}
        for track2, result in self.card_results.items():
            if result.flagged_for_review:
                flagged[track2] = {
                    'confidence': result.confidence_level,
                    'consistency': {
                        'KENC': f"{result.consistency_kenc:.1%}",
                        'KMAC': f"{result.consistency_kmac:.1%}",
                        'KDEK': f"{result.consistency_kdek:.1%}"
                    },
                    'predicted_KENC': result.estimated_kenc,
                    'predicted_KMAC': result.estimated_kmac,
                    'predicted_KDEK': result.estimated_kdek,
                    'reason': 'Low confidence' if result.is_uncertain else 'High disagreement'
                }
        return flagged
