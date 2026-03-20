"""
CPA (Correlation Power Analysis) Attack for 3DES using RAW TRACES
Recovers keys by correlating power traces with hypothetical S-Box outputs
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from scipy.stats import pearsonr
import sys
from collections import defaultdict

# Add pipeline-code to path
sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE, generate_round_keys

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

def load_raw_traces(input_dir, trace_files, max_traces=None):
    """
    Load raw power traces from CSV files with trace_data and ATC columns
    
    Args:
        input_dir: Directory containing trace CSVs
        trace_files: List of CSV filenames to load
        max_traces: Limit total traces to this many
    
    Returns:
        traces: Numpy array of shape (num_traces, num_samples)
        atc_bytes: Numpy array of shape (num_traces, 8) with padded ATC values
        ground_truths: Dict with KENC, KMAC, KDEK keys
    """
    input_path = Path(input_dir)
    all_traces = []
    all_atc_bytes = []
    ground_truths = {'KENC': None, 'KMAC': None, 'KDEK': None}
    
    logger.info(f"Loading raw traces from {input_dir}")
    
    for csv_file in trace_files:
        csv_path = input_path / csv_file
        if not csv_path.exists():
            logger.warning(f"File not found: {csv_path}")
            continue
        
        logger.info(f"  Loading {csv_file}...")
        df = pd.read_csv(csv_path)
        
        logger.info(f"    Loaded {len(df)} traces")
        
        # Extract from first row to get ground truth (should be constant)
        if 'T_DES_KENC' in df.columns:
            ground_truths['KENC'] = df.iloc[0]['T_DES_KENC']
            ground_truths['KMAC'] = df.iloc[0]['T_DES_KMAC']
            ground_truths['KDEK'] = df.iloc[0]['T_DES_KDEK']
        
        # Parse trace_data (comma-separated floats) and ATC
        for idx, row in df.iterrows():
            # Parse trace data
            try:
                trace_str = str(row['trace_data'])
                trace_samples = [float(x.strip()) for x in trace_str.split(',')]
                all_traces.append(trace_samples)
            except Exception as e:
                logger.warning(f"Failed to parse trace {idx}: {e}")
                continue
            
            # Parse ATC (hex string like "7A CD")
            try:
                atc_str = str(row['ATC']).strip()
                # Split hex string and convert to bytes
                atc_hex_parts = atc_str.split()
                # Convert to integers and pad to 8 bytes
                atc_vals = [0] * 8  # Initialize with 6 zeros (first 6 bytes)
                for i, hex_byte in enumerate(atc_hex_parts):
                    atc_vals[6 + i] = int(hex_byte, 16)  # Last 2 bytes
                all_atc_bytes.append(atc_vals)
            except Exception as e:
                logger.warning(f"Failed to parse ATC {idx}: {e}")
                all_atc_bytes.append([0] * 8)
        
        if max_traces and len(all_traces) >= max_traces:
            logger.info(f"  Reached max_traces limit ({max_traces})")
            break
    
    if not all_traces:
        raise ValueError("No traces loaded")
    
    # Limit to max_traces if needed
    if max_traces and len(all_traces) > max_traces:
        all_traces = all_traces[:max_traces]
        all_atc_bytes = all_atc_bytes[:max_traces]
    
    # Convert to numpy arrays
    traces = np.array(all_traces, dtype=np.float32)
    atc_bytes = np.array(all_atc_bytes, dtype=np.uint8)
    
    logger.info(f"Total traces loaded: {traces.shape[0]}")
    logger.info(f"Samples per trace: {traces.shape[1]}")
    logger.info(f"ATC shape: {atc_bytes.shape}")
    logger.info(f"Ground truths: {ground_truths}")
    
    return traces, atc_bytes, ground_truths

def cpa_attack_sbox_byte(traces, plaintexts_bytes, sbox_idx, round_key, max_traces=None):
    """
    CPA attack on single S-Box to recover key byte
    
    Uses Hamming weight power model and Pearson correlation
    
    Args:
        traces: Power traces (num_traces, num_samples)
        plaintexts_bytes: Plaintext bytes (num_traces, 8)
        sbox_idx: S-Box index (0-7)
        round_key: 6-byte round key for this position
        max_traces: Limit traces to this many (for speed)
    
    Returns:
        results: Dict with best_key, max_correlation, all_correlations
    """
    
    num_traces = traces.shape[0]
    if max_traces:
        num_traces = min(max_traces, num_traces)
        traces = traces[:num_traces]
        plaintexts_bytes = plaintexts_bytes[:num_traces]
    
    # Compute power model for this S-Box position
    # Power model: Hamming weight of S-Box output
    
    logger.info(f"    Attacking S-Box {sbox_idx + 1} with {num_traces} traces...")
    
    correlations = {}
    
    # Try all 64 possible 6-bit values
    for hypothesis in range(64):
        # Compute expected power for each trace
        expected_power = np.zeros(num_traces, dtype=np.float32)
        
        for trace_idx in range(num_traces):
            # Get plaintext block
            pt_bytes = plaintexts_bytes[trace_idx]
            plaintext = int.from_bytes(pt_bytes, 'big')
            
            # Apply initial permutation
            L_R = apply_permutation(plaintext, IP, width=64)
            R = L_R & 0xFFFFFFFF
            
            # Expand R for first round
            R_expanded = apply_permutation(R, E_TABLE, width=32)
            
            # Extract 6-bit input for this S-Box
            # The hypothesis IS the 6-bit value after XOR with round key
            shift = 42 - (sbox_idx * 6)
            
            # Apply S-Box
            sbox_output = des_sbox_output(sbox_idx, hypothesis)
            
            # Power model: Hamming weight
            expected_power[trace_idx] = hamming_weight(sbox_output)
        
        # Correlate with actual power
        # Use differential power consumption: max - min for each sample
        actual_power_agg = np.sum(traces, axis=1)
        
        try:
            corr, pval = pearsonr(expected_power, actual_power_agg)
            correlations[hypothesis] = corr
        except Exception as e:
            correlations[hypothesis] = 0
    
    # Get best hypothesis
    best_hypothesis = max(correlations, key=lambda x: abs(correlations[x]))
    max_correlation = abs(correlations[best_hypothesis])
    
    return {
        'best_key': best_hypothesis,
        'max_correlation': max_correlation,
        'all_correlations': correlations
    }

def cpa_attack_key(traces, atc_bytes, ground_truths, key_type='KENC', max_traces_per_sbox=None):
    """
    Full CPA attack to recover key
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"CPA ATTACK FOR {key_type}")
    logger.info(f"{'='*70}\n")
    
    num_traces = traces.shape[0]
    
    gt_key_hex = ground_truths.get(key_type)
    if not gt_key_hex:
        logger.warning(f"No ground truth for {key_type}")
        return None
    
    logger.info(f"Ground truth {key_type}: {gt_key_hex}")
    logger.info(f"Using {num_traces} traces for CPA attack\n")
    
    # Attack each S-Box
    recovered_bits = []
    correlations_per_sbox = []
    
    for sbox_idx in range(8):
        result = cpa_attack_sbox_byte(
            traces, 
            atc_bytes, 
            sbox_idx, 
            round_key=None,
            max_traces=max_traces_per_sbox
        )
        
        recovered_bits.append(result['best_key'])
        correlations_per_sbox.append(result['max_correlation'])
        
        # Show top correlations
        top_corrs = sorted(
            result['all_correlations'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        logger.info(f"    Best key: {result['best_key']:2d}, Correlation: {result['max_correlation']:+.6f}")
        logger.info(f"    Top 3 hypotheses:")
        for hyp, corr in top_corrs[:3]:
            logger.info(f"      {hyp:2d}: {corr:+.6f}")
    
    logger.info(f"\nRecovered S-Box hypotheses: {recovered_bits}")
    logger.info(f"Correlation scores: {[f'{c:.4f}' for c in correlations_per_sbox]}")
    
    return {
        'key_type': key_type,
        'ground_truth': gt_key_hex,
        'sbox_hypotheses': recovered_bits,
        'correlations': correlations_per_sbox,
        'num_traces': num_traces
    }

def main():
    """Main CPA attack pipeline using raw traces"""
    
    # Configuration
    input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
    trace_files = [
        'traces_data_1000T_1.csv',
        'traces_data_2000T_2.csv',
        'traces_data_2000T_3.csv',
        'traces_data_2000T_4.csv',
        'traces_data_3000T_5.csv',
    ]
    
    # Load raw traces
    try:
        traces, atc_bytes, ground_truths = load_raw_traces(input_dir, trace_files, max_traces=10000)
    except Exception as e:
        logger.error(f"Failed to load traces: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info(f"\nTrace statistics:")
    logger.info(f"  Min: {traces.min():.4f}")
    logger.info(f"  Max: {traces.max():.4f}")
    logger.info(f"  Mean: {traces.mean():.4f}")
    logger.info(f"  Std: {traces.std():.4f}")
    
    # Run CPA for each key type
    results = {}
    
    for key_type in ['KENC', 'KMAC', 'KDEK']:
        if ground_truths.get(key_type):
            # Use subset of traces for speed (CPA is slow)
            result = cpa_attack_key(traces, atc_bytes, ground_truths, key_type, max_traces_per_sbox=5000)
            if result:
                results[key_type] = result
        else:
            logger.warning(f"No ground truth for {key_type}, skipping")
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL CPA ATTACK RESULTS")
    logger.info(f"{'='*70}\n")
    
    for key_type, result in results.items():
        logger.info(f"{key_type}:")
        logger.info(f"  Ground Truth:     {result['ground_truth']}")
        logger.info(f"  S-Box Hypotheses: {result['sbox_hypotheses']}")
        logger.info(f"  Correlation Mean: {np.mean(result['correlations']):.6f}")
        logger.info(f"  Correlation Std:  {np.std(result['correlations']):.6f}")
    
    logger.info("\nCPA attack complete!")
    
    return results

if __name__ == '__main__':
    main()
