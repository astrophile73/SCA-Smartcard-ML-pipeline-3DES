"""
Full CPA Attack - Run on all keys with all traces and verify results
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pipeline-code'))

from src.crypto import des_sbox_output, apply_permutation, IP, E_TABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def hamming_weight(x):
    return bin(x).count('1')

def load_all_traces(input_dir):
    """Load all 10,000 traces"""
    input_path = Path(input_dir)
    trace_files = [
        'traces_data_1000T_1.csv',
        'traces_data_2000T_2.csv',
        'traces_data_2000T_3.csv',
        'traces_data_2000T_4.csv',
        'traces_data_3000T_5.csv',
    ]
    
    all_traces = []
    all_atc_bytes = []
    ground_truths = {}
    
    for csv_file in trace_files:
        csv_path = input_path / csv_file
        logger.info(f"Loading {csv_file}...")
        
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            # Parse trace
            trace_str = str(row['trace_data'])
            samples = np.array([float(x.strip()) for x in trace_str.split(',')], dtype=np.float32)
            all_traces.append(samples)
            
            # Parse ATC
            atc_str = str(row['ATC']).strip()
            atc_hex_parts = atc_str.split()
            atc_vals = [0] * 8
            for i, hex_byte in enumerate(atc_hex_parts):
                atc_vals[6 + i] = int(hex_byte, 16)
            all_atc_bytes.append(atc_vals)
        
        # Get ground truth from first row
        if not ground_truths:
            ground_truths['KENC'] = df.iloc[0]['T_DES_KENC']
            ground_truths['KMAC'] = df.iloc[0]['T_DES_KMAC']
            ground_truths['KDEK'] = df.iloc[0]['T_DES_KDEK']
    
    traces = np.array(all_traces, dtype=np.float32)
    atc_bytes = np.array(all_atc_bytes, dtype=np.uint8)
    
    logger.info(f"Total traces: {len(traces)}, shape: {traces.shape}")
    
    return traces, atc_bytes, ground_truths

def cpa_attack_sbox(traces, atc_bytes, sbox_idx, max_traces=None):
    """CPA attack on single S-Box"""
    
    num_traces = traces.shape[0]
    if max_traces:
        num_traces = min(max_traces, num_traces)
        traces = traces[:num_traces]
        atc_bytes = atc_bytes[:num_traces]
    
    correlations = {}
    
    for hyp in range(64):
        expected_power = np.zeros(num_traces, dtype=np.float32)
        
        for t_idx in range(num_traces):
            pt_bytes = atc_bytes[t_idx]
            plaintext64 = int.from_bytes(pt_bytes, 'big')
            L_R = apply_permutation(plaintext64, IP, width=64)
            R = L_R & 0xFFFFFFFF
            R_expanded = apply_permutation(R, E_TABLE, width=32)
            shift = 42 - (sbox_idx * 6)
            sbox_input_fromR = (R_expanded >> shift) & 0x3F
            sbox_input = sbox_input_fromR ^ hyp
            sbox_output = des_sbox_output(sbox_idx, sbox_input)
            expected_power[t_idx] = hamming_weight(sbox_output)
        
        actual_power = np.mean(traces, axis=1)
        
        try:
            if np.std(expected_power) > 0 and np.std(actual_power) > 0:
                corr, _ = pearsonr(expected_power, actual_power)
                correlations[hyp] = corr
            else:
                correlations[hyp] = 0
        except:
            correlations[hyp] = 0
    
    best_hyp = max(correlations, key=lambda x: abs(correlations[x]))
    max_corr = abs(correlations[best_hyp])
    
    return best_hyp, max_corr, correlations

def cpa_attack_key(traces, atc_bytes, key_type):
    """Attack all S-Boxes for a key"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"CPA ATTACK FOR {key_type}")
    logger.info(f"{'='*70}\n")
    
    recovered = []
    correlations_list = []
    
    for sbox_idx in range(8):
        logger.info(f"S-Box {sbox_idx+1}...")
        best_hyp, max_corr, _ = cpa_attack_sbox(traces, atc_bytes, sbox_idx, max_traces=None)
        recovered.append(best_hyp)
        correlations_list.append(max_corr)
        logger.info(f"  -> hypothesis={best_hyp:2d}, corr={max_corr:+.4f}")
    
    return recovered, correlations_list

def main():
    input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
    
    logger.info("Loading all 10,000 traces...")
    traces, atc_bytes, ground_truths = load_all_traces(input_dir)
    
    logger.info(f"Traces shape: {traces.shape}")
    logger.info(f"Ground truths: {ground_truths}")
    
    # Run CPA for each key
    results = {}
    for key_type in ['KENC', 'KMAC', 'KDEK']:
        recovered, corrs = cpa_attack_key(traces, atc_bytes, key_type)
        results[key_type] = {
            'recovered_bits': recovered,
            'correlations': corrs,
            'ground_truth': ground_truths[key_type]
        }
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}\n")
    
    for key_type, result in results.items():
        logger.info(f"{key_type}:")
        logger.info(f"  Ground truth:    {result['ground_truth']}")
        logger.info(f"  Recovered bits:  {result['recovered_bits']}")
        logger.info(f"  Mean correlation: {np.mean(result['correlations']):.4f}")
        logger.info(f"  Correlations:     {[f'{c:.4f}' for c in result['correlations']]}")

if __name__ == '__main__':
    main()
