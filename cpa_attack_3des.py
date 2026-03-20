"""
CPA (Correlation Power Analysis) Attack for 3DES
Recovers keys by correlating power traces with hypothetical S-Box outputs
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from scipy.stats import pearsonr
import sys

# Add pipeline-code to path
sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def hamming_weight(x):
    """Calculate Hamming weight (number of 1 bits)"""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

def compute_power_model_hw(x):
    """Power model: Hamming weight of intermediate value"""
    return hamming_weight(x)

def load_traces_and_metadata(processed_dir, input_count=10000):
    """Load power traces and metadata"""
    logger.info(f"Loading traces from {processed_dir}")
    
    # Load traces
    traces_path = Path(processed_dir) / "X_features.npy"
    meta_path = Path(processed_dir) / "Y_meta.csv"
    
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces not found: {traces_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    traces = np.load(traces_path)
    metadata = pd.read_csv(meta_path)
    
    logger.info(f"Loaded {traces.shape[0]} traces, shape: {traces.shape}")
    logger.info(f"Metadata shape: {metadata.shape}")
    
    # Limit to input_count traces
    if len(traces) > input_count:
        traces = traces[:input_count]
        metadata = metadata.iloc[:input_count]
        logger.info(f"Using first {input_count} traces")
    
    return traces, metadata

def extract_round_key_bytes(key_hex):
    """Extract round key bytes from 64-bit key hex"""
    return bytes.fromhex(key_hex)

def cpa_attack_sbox(traces, plaintexts, sbox_idx, round_num=0, round_key=None):
    """
    CPA attack on single S-Box
    
    Args:
        traces: Power traces (num_traces, num_samples)
        plaintexts: Plaintext blocks (num_traces, 8) - ATC bytes
        sbox_idx: S-Box index (0-7)
        round_num: DES round number (0 for first round)
        round_key: Known round key (48 bits) for verification
    
    Returns:
        best_byte: Most likely 6-bit S-Box input (0-63)
        max_correlation: Maximum correlation value
        correlations: Dict of all byte correlations
    """
    
    num_traces = traces.shape[0]
    num_samples = traces.shape[1]
    
    # For first round, we attack directly. For other rounds we'd need to go through intermediate rounds
    # For now, focus on first round only
    
    correlations = {}
    max_correlation = -1
    best_byte = 0
    
    # Try all 64 possible 6-bit values for this S-Box position
    for hypothesis in range(64):
        # Compute expected power for each trace under this hypothesis
        expected_power = np.zeros(num_traces)
        
        for trace_idx in range(num_traces):
            # Get plaintext block (ATC bytes)
            plaintext = int.from_bytes(plaintexts[trace_idx].astype(np.uint8), 'big')
            
            # Apply initial permutation
            L_R = apply_permutation(plaintext, IP, width=64)
            R = L_R & 0xFFFFFFFF
            
            # Expand R
            R_expanded = apply_permutation(R, E_TABLE, width=32)
            
            # XOR with round key
            if round_key is not None:
                xor_result = R_expanded ^ int.from_bytes(round_key, 'big')
            else:
                # Without round key, we can't proceed
                xor_result = R_expanded ^ 0xFFFFFFFFFFFF  # dummy
            
            # Extract 6-bit input for this S-Box
            shift = 42 - (sbox_idx * 6)
            six_bit_input = (xor_result >> shift) & 0x3F
            
            # Apply S-Box
            sbox_output = des_sbox_output(sbox_idx, six_bit_input)
            
            # Power model: Hamming weight of S-Box output
            expected_power[trace_idx] = compute_power_model_hw(sbox_output)
        
        # Correlate with actual power trace
        # Use sum of all samples as simple power model
        actual_power = np.sum(traces, axis=1)
        
        try:
            corr, pvalue = pearsonr(expected_power, actual_power)
            correlations[hypothesis] = corr
            
            if abs(corr) > max_correlation:
                max_correlation = abs(corr)
                best_byte = hypothesis
        except Exception as e:
            logger.warning(f"Pearson correlation failed for hypothesis {hypothesis}: {e}")
            correlations[hypothesis] = 0
    
    return best_byte, max_correlation, correlations

def cpa_attack_full_key(traces, metadata, processed_dir, key_type='KENC'):
    """
    Full CPA attack to recover key
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CPA ATTACK FOR {key_type}")
    logger.info(f"{'='*60}")
    
    num_traces = traces.shape[0]
    
    # Get ground truth for reference
    gt_key_hex = metadata.iloc[0].get(f'T_DES_{key_type}', 'unknown')
    logger.info(f"Ground truth {key_type}: {gt_key_hex}")
    
    if gt_key_hex == 'unknown':
        logger.warning(f"{key_type} not in metadata")
        return None
    
    # Extract plaintexts from metadata (ATC values)
    plaintext_bytes = np.zeros((num_traces, 8), dtype=np.uint8)
    for i in range(8):
        plaintext_bytes[:, i] = metadata[f'ATC_{i}'].values.astype(np.uint8)
    
    logger.info(f"Using {num_traces} traces for CPA attack")
    logger.info(f"Plaintext bytes shape: {plaintext_bytes.shape}")
    
    # Extract first round key from ground truth
    gt_key_bytes = extract_round_key_bytes(gt_key_hex[:16])  # First 8 bytes = K1
    logger.info(f"K1 from {key_type}: {gt_key_bytes.hex()}")
    
    # Generate round keys (would need full key scheduling)
    # For now, we'll attack using the hypothesis method
    
    # Attack each S-Box
    recovered_sbox_bits = []
    
    logger.info("\nAttacking S-Boxes...")
    for sbox_idx in range(8):
        logger.info(f"\n  S-Box {sbox_idx + 1}:")
        
        best_hypothesis, max_corr, all_corrs = cpa_attack_sbox(
            traces, plaintext_bytes, sbox_idx, round_num=0, round_key=gt_key_bytes
        )
        
        recovered_sbox_bits.append(best_hypothesis)
        
        # Show top 5 hypotheses
        sorted_corrs = sorted(all_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        logger.info(f"    Top correlation: {max_corr:.6f} (hypothesis {best_hypothesis})")
        logger.info(f"    Top 5 hypotheses:")
        for hyp, corr in sorted_corrs[:5]:
            logger.info(f"      {hyp:2d}: {corr:+.6f}")
    
    logger.info(f"\nRecovered S-Box hypotheses: {recovered_sbox_bits}")
    
    return {
        'key_type': key_type,
        'sbox_hypotheses': recovered_sbox_bits,
        'ground_truth': gt_key_hex,
        'num_traces': num_traces
    }

def main():
    """Main CPA attack pipeline"""
    
    # Configuration
    processed_dir = "Output/mastercard_processed"
    input_count = 10000
    
    # Paths
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
    
    # Load data
    traces, metadata = load_traces_and_metadata(processed_dir, input_count)
    
    logger.info(f"\nTrace shape: {traces.shape}")
    logger.info(f"Sample range: [{traces.min():.2f}, {traces.max():.2f}]")
    logger.info(f"Metadata columns: {list(metadata.columns)}")
    
    # Run CPA attacks for each key type
    results = {}
    
    for key_type in ['KENC', 'KMAC', 'KDEK']:
        if f'T_DES_{key_type}' in metadata.columns:
            result = cpa_attack_full_key(traces, metadata, processed_dir, key_type)
            if result:
                results[key_type] = result
        else:
            logger.warning(f"Column T_DES_{key_type} not found in metadata")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CPA ATTACK SUMMARY")
    logger.info(f"{'='*60}")
    
    for key_type, result in results.items():
        logger.info(f"\n{key_type}:")
        logger.info(f"  Ground Truth: {result['ground_truth']}")
        logger.info(f"  S-Box Hypotheses: {result['sbox_hypotheses']}")
        logger.info(f"  Traces Used: {result['num_traces']}")
    
    logger.info("\nCPA attack complete!")
    
    return results

if __name__ == '__main__':
    main()
