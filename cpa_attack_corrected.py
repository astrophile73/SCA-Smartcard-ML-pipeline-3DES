"""
Corrected CPA Attack for 3DES
Uses proper intermediate value computation for first round
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def hamming_weight(x):
    """Calculate Hamming weight"""
    return bin(x).count('1')

def load_traces_simple(input_dir, max_traces=1000):
    """Load subset of traces for faster testing"""
    input_path = Path(input_dir)
    csv_path = input_path / 'traces_data_1000T_1.csv'
    
    logger.info(f"Loading traces from {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_traces)
    
    traces = []
    atc_bytes = []
    
    for idx, row in df.iterrows():
        # Parse trace
        trace_str = str(row['trace_data'])
        samples = np.array([float(x.strip()) for x in trace_str.split(',')], dtype=np.float32)
        traces.append(samples)
        
        # Parse ATC
        atc_str = str(row['ATC']).strip()
        atc_hex_parts = atc_str.split()
        atc_vals = [0] * 8
        for i, hex_byte in enumerate(atc_hex_parts):
            atc_vals[6 + i] = int(hex_byte, 16)
        atc_bytes.append(atc_vals)
    
    traces = np.array(traces, dtype=np.float32)
    atc_bytes = np.array(atc_bytes, dtype=np.uint8)
    
    gt_kenc = df.iloc[0]['T_DES_KENC']
    
    logger.info(f"Loaded {len(traces)} traces, shape: {traces.shape}")
    logger.info(f"ATC shape: {atc_bytes.shape}")
    logger.info(f"Ground truth KENC: {gt_kenc}")
    
    return traces, atc_bytes, gt_kenc

def cpa_attack_sbox_byte_corrected(traces, atc_bytes, sbox_idx, round_key_byte_idx, max_traces=None):
    """
    Corrected CPA attack on single S-Box

    For each 6-bit key hypothesis:
    - Compute S-Box input = plaintext_derived_bits XOR key_bits
    - Compute S-Box output
    - Model power as Hamming weight
    - Correlate with actual power
    
    Args:
        sbox_idx: S-Box index (0-7)
        round_key_byte_idx: Which byte of the round key (0-5 for 48-bit RK)
    """
    
    num_traces = traces.shape[0]
    if max_traces:
        num_traces = min(max_traces, num_traces)
        traces = traces[:num_traces]
        atc_bytes = atc_bytes[:num_traces]
    
    logger.info(f"Attacking S-Box {sbox_idx+1}, byte {round_key_byte_idx} with {num_traces} traces...")
    
    # Compute power model for each hypothesis
    correlations = {}
    
    for hyp in range(64):  # 6-bit hypothesis
        # For each trace, compute expected S-Box output
        expected_power = np.zeros(num_traces, dtype=np.float32)
        
        for t_idx in range(num_traces):
            # Get plaintext
            pt_bytes = atc_bytes[t_idx]
            plaintext64 = int.from_bytes(pt_bytes, 'big')
            
            # Apply initial permutation
            L_R = apply_permutation(plaintext64, IP, width=64)
            R = L_R & 0xFFFFFFFF
            
            # Expand right half
            R_expanded = apply_permutation(R, E_TABLE, width=32)
            
            # The R_expanded is 48 bits, divided into 8 groups of 6 bits for each S-Box
            # Extract 6-bit portion for this S-Box
            # S-Boxes consume bits from left to right: S1=47-42, S2=41-36, ..., S8=5-0
            shift = 42 - (sbox_idx * 6)
            sbox_input_fromR = (R_expanded >> shift) & 0x3F
            
            # Apply hypothesis key bits (XOR with key hypothesis)
            sbox_input = sbox_input_fromR ^ hyp
            
            # Apply S-Box
            sbox_output = des_sbox_output(sbox_idx, sbox_input)
            
            # Power model: Hamming weight of 4-bit S-Box output
            expected_power[t_idx] = hamming_weight(sbox_output)
        
        # Correlate with actual power
        # Use sum of all samples as aggregate power
        actual_power = np.mean(traces, axis=1)  # Average power per trace
        
        try:
            # Compute Pearson correlation
            if np.std(expected_power) > 0 and np.std(actual_power) > 0:
                corr, pval = pearsonr(expected_power, actual_power)
                correlations[hyp] = corr
            else:
                correlations[hyp] = 0
        except:
            correlations[hyp] = 0
    
    # Find best hypothesis
    best_hyp = max(correlations, key=lambda x: abs(correlations[x]))
    max_corr = abs(correlations[best_hyp])
    
    # Show top hypotheses
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info(f"  Top 5:")
    for hyp, corr in sorted_corrs[:5]:
        logger.info(f"    {hyp:2d}: {corr:+.6f}")
    
    return best_hyp, max_corr, correlations

def cpa_attack_first_round(traces, atc_bytes, gt_key_hex):
    """Attack first round to recover key hypotheses"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"FIRST-ROUND CPA ATTACK")
    logger.info(f"Ground truth: {gt_key_hex}")
    logger.info(f"{'='*70}\n")
    
    recovered = []
    
    # Attack each of 8 S-Boxes
    for sbox_idx in range(8):
        logger.info(f"\nS-Box {sbox_idx+1}:")
        
        best_hyp, best_corr, corrs = cpa_attack_sbox_byte_corrected(
            traces, atc_bytes, sbox_idx, sbox_idx, max_traces=5000
        )
        
        recovered.append(best_hyp)
        logger.info(f"  Best hypothesis: {best_hyp} (correlation: {best_corr:+.6f})")
    
    logger.info(f"\nRecovered S-Box key bits: {recovered}")
    
    return recovered

def main():
    input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
    
    # Load traces
    traces, atc_bytes, gt_kenc = load_traces_simple(input_dir, max_traces=5000)
    
    logger.info(f"\nTrace statistics:")
    logger.info(f"  Min: {traces.min():.6f}")
    logger.info(f"  Max: {traces.max():.6f}")
    logger.info(f"  Mean: {traces.mean():.6f}")
    logger.info(f"  Std: {traces.std():.6f}")
    
    # Run CPA
    recovered = cpa_attack_first_round(traces, atc_bytes, gt_kenc)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"CPA ATTACK COMPLETE")
    logger.info(f"Recovered S-Box bits: {recovered}")
    logger.info(f"Expected (from ground truth key): ...")
    logger.info(f"{'='*70}")

if __name__ == '__main__':
    main()
