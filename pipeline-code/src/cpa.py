import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm
from src.crypto import des_sbox_output, hamming_weight
from src.utils import setup_logger

logger = setup_logger("cpa")

class CPAEngine:
    def __init__(self, trace_len: int):
        self.trace_len = trace_len
        self.best_guess = None
        self.correlations = None

    def run_cpa(self, traces: np.ndarray, plaintexts: np.ndarray, known_key: Optional[bytes] = None) -> Dict:
        """
        Run CPA to recover DES Key (K1).
        plaintexts: (N, 8) array of uint8
        """
        n_traces = traces.shape[0]
        n_samples = traces.shape[1]
        
        # DES has 8 S-Boxes
        # We process each S-Box
        
        recovered_key_fragments = []
        full_correlations = {}
        
        # Precompute traces statistics for correlation
        # (X - mean(X))
        traces_mean = np.mean(traces, axis=0)
        traces_centered = traces - traces_mean
        # For denominator of correlation: sum((X-mean)^2)
        traces_var = np.sum(traces_centered ** 2, axis=0)
        
        # Avoid division by zero
        traces_var[traces_var == 0] = 1.0
        
        for sbox_idx in range(8):
            logger.info(f"Attacking S-Box {sbox_idx + 1}...")
            
            max_corr = -1
            best_k = 0
            
            # Key guess: 6 bits = 64 possibilities
            # But the 6 bits come from the 56-bit key. 
            # We recover the 6-bit subkey input to the S-Box.
            
            # Correlate
            k_corrs = []
            
            for k_guess in range(64):
                # Compute intermediate: HW(SBox(P_sub XOR k_guess))
                # We need to map plaintext bits to SBox input bits.
                # Standard DES Expansion involves permutation.
                # Simplification: Assume we have the extracted relevant 6 bits of P for this S-Box.
                # We need a mapper: (N, 8) -> (N, 6 bits) for SBox i.
                
                # Wait, getting the correct 6 bits of Plaintext requires the Expansion function E.
                # P (32 bits R0) -> E(R0) (48 bits).
                # SBox i takes 6 bits effectively from R0 (which is part of P).
                # Actually, R0 = R part of IP(Block).
                # If we assume standard DES...
                
                # Let's implement full DES round 1 Logic or use a simplified hypothesis.
                # Usually: bits of P used for S1: 32, 1, 2, 3, 4, 5 (1-based from R0).
                
                # BUT, wait. Is it 3DES or DES?
                # 3DES uses DES. K1 CPA is same as DES CPA.
                
                # I will assume standard DES expansion.
                # Need `get_sbox_input_bits(plaintext, sbox_idx)` helper.
                
                # Let's calculate leakage
                leakage = np.zeros(n_traces)
                
                # Optimized vectorization possible?
                # For this limited scope, loop is fine or list comp.
                
                # To be correct, we need the exact bits.
                # I'll add `get_p_bits` method.
                pass
                
        # For now, returning stub
        return {}
        
    def find_pois(self, traces: np.ndarray, plaintexts: np.ndarray, key: bytes) -> np.ndarray:
        """
        Find POIs using the known key K1.
        Returns indices of high correlation.
        """
        # key needs to be 8 bytes (64 bits with parity)
        # We attack Round 1 S-Boxes.
        
        pass

def bit(x, i):
    return (x >> i) & 1

# Expansion table for DES (1-based index from R0 32-bits)
E_TABLE = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]

def get_expansion_input(r0_block: int) -> int:
    # r0_block is 32-bit integer
    out = 0
    for i in range(48):
        # E_TABLE is 1-based
        pos = E_TABLE[i] - 1
        # bit at pos (0..31) from MSB or LSB? 
        # DES usually bit 1 is MSB.
        # But integers usually bit 0 is LSB.
        # Let's assume standard integer bit numbering: bit 0 is LSB.
        # Standard DES: bit 1 is MSB.
        # So pos 0 (DES 1) is bit 31 (int).
        
        b = (r0_block >> (31 - pos)) & 1
        out |= (b << (47 - i))
    return out
