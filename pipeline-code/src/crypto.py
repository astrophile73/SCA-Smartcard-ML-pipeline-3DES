
import logging
from src.pyDes import des, triple_des

# LOGGING
logger = logging.getLogger("crypto")

# =============================================================================
# TABLES FROM BLACKRABBIT pyDes.py
# Copied to ensure 100% logic migration as requested.
# All tables are 0-indexed.
# =============================================================================

# Permutation and translation tables for DES
# 0-indexed indices into the 64-bit block
_PC1 = [56, 48, 40, 32, 24, 16,  8,
      0, 57, 49, 41, 33, 25, 17,
      9,  1, 58, 50, 42, 34, 26,
     18, 10,  2, 59, 51, 43, 35,
     62, 54, 46, 38, 30, 22, 14,
      6, 61, 53, 45, 37, 29, 21,
     13,  5, 60, 52, 44, 36, 28,
     20, 12,  4, 27, 19, 11,  3
]

# permuted choice key (table 2)
# 0-indexed indices into the 56-bit C+D block
_PC2 = [
    13, 16, 10, 23,  0,  4,
     2, 27, 14,  5, 20,  9,
    22, 18, 11,  3, 25,  7,
    15,  6, 26, 19, 12,  1,
    40, 51, 30, 36, 46, 54,
    29, 39, 50, 44, 32, 47,
    43, 48, 38, 55, 33, 52,
    45, 41, 49, 35, 28, 31
]

# number left rotations of pc1
_LEFT_ROTATIONS = [
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
]

# Standard S-boxes (S1-S8) - Copied from pyDes
# Each S-box is 64 elements (4 rows * 16 cols flattened)
_SBOX = [
    # S1
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
     0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
     4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
     15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
    # S2
    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
     3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
     13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
    # S3
    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
     13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
     13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
      1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
    # S4
    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
     13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
     10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
     3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
    # S5
    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
     14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
     4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
     11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
    # S6
    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
     10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
     9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
     4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
    # S7
    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
     13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
      1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
     6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
    # S8
    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
     1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
     7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
     2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
]

def get_sbox_input_candidates(sbox_idx, four_bit_output):
    """
    Finds all 6-bit inputs that produce the given 4-bit S-box output.
    Returns a list of 4 integers (0-63).
    """
    candidates = []
    for i in range(64):
        if des_sbox_output(sbox_idx, i) == four_bit_output:
            candidates.append(i)
    return candidates

# Expansion table for turning 32 bit blocks into 48 bits
# 0-indexed indices into the 32-bit block
_EXPANSION_TABLE = [
    31,  0,  1,  2,  3,  4,
     3,  4,  5,  6,  7,  8,
      7,  8,  9, 10, 11, 12,
    11, 12, 13, 14, 15, 16,
    15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24,
    23, 24, 25, 26, 27, 28,
    27, 28, 29, 30, 31,  0
]

# initial permutation IP
# 0-indexed indices into the 64-bit block
_IP = [57, 49, 41, 33, 25, 17, 9,  1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7,
    56, 48, 40, 32, 24, 16, 8,  0,
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6
]

# --- LEGACY ALIASES & HELPERS (Required by feature_eng.py) ---
# Aliases to match expected export names
E_TABLE = _EXPANSION_TABLE
# Note: IP table in feature_eng might expect 1-based indexing if it was using old crypto.
# But generally standard algos use 1-based in docs, 0-based in code.
# Let's assume 0-based for now as it's common in Python implementations.
IP = _IP

def hamming_weight(x):
    """Calculates Hamming Weight of an integer."""
    return bin(x).count('1')

def apply_permutation(data, table, width=None):
    """
    Applies a permutation table to a bit sequence.
    Args:
        data: input data (often int or list of bits)
        table: standard permutation table (0-based indices)
        width: Optional explicit bit width of the input data.
    Returns:
        Permuted result.
    """
    # If data is int, convert to bits (assuming 64 bit for IP usually)
    # But feature_eng context suggests usage.
    # If table is IP (64->64), data usually 64-bit int.
    # If table is E (32->48), data usually 32-bit int.
    
    # We'll support int->int for basic usage
    if isinstance(data, int):
        # Determine bit length. IP needs 64 source, E needs 32 source.
        # But we can just handle generic.
        # Let's look at table max index to guess input size if strict.
        # Max index in IP is 63 -> 64 bits.
        
        # Simple extraction
        permuted = 0
        target_len = len(table)
        
        # We need to be careful about bit ordering (MSB 0 vs LSB 0).
        # pyDes treats index 0 as MSB.
        # Let's assume standard DES big-endian bit numbering where bit 1 is MSB.
        # In Python int: MSB is high value.
        # If we have 64 bit int X. Bit 0 (DES) is the MSB (2^63).
        
        if width is not None:
             input_len = width
        else:
            # We need a size.
            input_len = max(table) + 1 # Rough guess
            if input_len <= 32: input_len = 32
            elif input_len <= 48: input_len = 48
            elif input_len <= 64: input_len = 64
        
        for i, idx in enumerate(table):
            # idx is the index of the source bit (0 is MSB).
            # To extract bit at index 'idx' from MSB:
            # shift = (input_len - 1) - idx
            shift = (input_len - 1) - idx
            rotation = (data >> shift) & 1
            
            # To set bit at index 'i' in result (0 is MSB):
            # pos = (target_len - 1) - i
            pos = (target_len - 1) - i
            permuted |= (rotation << pos)
            
        return permuted
    
    # If list of bits
    elif isinstance(data, list):
        return [data[i] for i in table]
    return 0

def des_sbox_output(sbox_idx, six_bit_input):
    """
    Computes S-Box output for a given 6-bit input.
    Args:
        sbox_idx: 0-7
        six_bit_input: integer 0-63
    Returns:
        4-bit integer output
    """
    # Extract row/col
    # Row: Bits 0 and 5 (MSB and LSB of 6-bit input)
    # Col: Bits 1-4
    # Note: six_bit_input is integer. Bit 0 is MSB? No in int 6 bit:
    # 5 4 3 2 1 0 (bit pos)
    # In DES 6-bit block b1 b2 b3 b4 b5 b6
    # Row = b1b6
    # Col = b2b3b4b5
    
    # int val: b1 is MSB (32), b6 is LSB (1)
    
    # Get bit 5 (value 32) -> b1
    b1 = (six_bit_input >> 5) & 1
    # Get bit 0 (value 1) -> b6
    b6 = six_bit_input & 1
    
    row = (b1 << 1) | b6
    col = (six_bit_input >> 1) & 0xF # middle 4 bits
    
    # pyDes SBOX is flattened [row 0][row 1][row 2][row 3] where each row is 16 elements.
    # Index = row * 16 + col
    return _SBOX[sbox_idx][row * 16 + col]


def generate_round_keys(key_bytes):
    """
    Generates standard DES round keys using pyDes logic.
    Returns list of 16 integers (48-bit each).
    """
    d = des(key_bytes)
    # pyDes stores keys as lists of bits (integers 0 or 1)
    # We need to convert them to 48-bit integers for the pipeline
    keys_int = []
    for k_bits in d.Kn:
        val = 0
        for b in k_bits:
            val = (val << 1) | b
        keys_int.append(val)
    return keys_int

def reconstruct_key_from_rk1(sbox_inputs):
    """
    Reconstructs the 64-bit DES key (with parity) from the 48-bit Round Key 1.
    Uses the inverse of PC1 and PC2 tables from pyDes to ensure exact migration.
    
    Args:
        sbox_inputs: list of 8 integers (0-63), representing the 6-bit inputs.
                     Wait, if it's the KEY CHUNKS, then it's 6-bit key fragments.
                     If it's SBox inputs, they are XORed with R.
                     Assuming these are the *recovered Round Key 1 6-bit chunks*.
    
    Returns:
        int: The reconstructed 64-bit key.
    """
    # 1. Flatten the 8 key chunks (6 bits each) into a 48-bit Round Key 1
    rk1_bits = []
    for val in sbox_inputs:
        val = int(val) # Explicitly cast to Python int to avoid numpy int64 overflow
        # Convert 6-bit integer to 6 bits, MSB first
        bits = [(val >> i) & 1 for i in range(5, -1, -1)]
        rk1_bits.extend(bits)
    
    # 2. Inverse PC2: Map 48 bits back to 56 bits (C1, D1)
    # PC2 drops 8 bits. We will initialize 56 bits to 0.
    # pyDes _PC2 indices map Input[Index] -> Output[0..47]
    # So Output[i] = Input[_PC2[i]]
    # Inverse: Input[_PC2[i]] = Output[i]
    c1d1 = [0] * 56
    for i in range(48):
        original_idx = _PC2[i] # Index in C1D1
        c1d1[original_idx] = rk1_bits[i]
        
    # 3. Inverse Shift: Map (C1, D1) back to (C0, D0)
    # Round 1 rotation is 1 bit LEFT.
    # Inverse is 1 bit RIGHT.
    shift = _LEFT_ROTATIONS[0] # 1
    
    c1 = c1d1[:28]
    d1 = c1d1[28:]
    
    # Right Rotate by 'shift'
    c0 = c1[-shift:] + c1[:-shift]
    d0 = d1[-shift:] + d1[:-shift]
    
    c0d0 = c0 + d0
    
    # 4. Inverse PC1: Map 56 bit (C0, D0) back to 64 bit Key
    # pyDes _PC1 indices map Key[Index] -> Output[0..55]
    # Output[i] = Key[_PC1[i]]
    # Inverse: Key[_PC1[i]] = Output[i]
    key_bits = [0] * 64
    for i in range(56):
        original_key_idx = _PC1[i]
        key_bits[original_key_idx] = c0d0[i]
        
    # 5. bits to int
    key_int = 0
    for b in key_bits:
        key_int = (key_int << 1) | b
        
    # 6. Correct Parity (Optional but good for valid keys)
    # Iterate bytes, set LSB to make odd parity
    final_key = 0
    for i in range(8):
        # Extract byte i (from MSB, so shift = (7-i)*8)
        shift = (7 - i) * 8
        byte_val = (key_int >> shift) & 0xFF
        
        # Check parity (count of 1s)
        # We need ODD parity.
        # If count is even, flip LSB.
        # But wait, LSB is the bit we don't know (it was 0 from our reconstruction).
        # Actually PC1 ignores LSB (bit 8, 16..).
        # Our reconstruction set them to 0.
        # So we just need to set LSB to 1 if the other 7 bits have even weight.
        # If other 7 bits have odd weight, LSB stays 0.
        
        # Get top 7 bits
        top_7 = byte_val & 0xFE
        w = bin(top_7).count('1')
        if w % 2 == 0:
            # Even weight, need 1 more for odd parity
            byte_val |= 1
        else:
            # Odd weight, already good (LSB is 0)
            byte_val &= 0xFE # Ensure LSB is 0
            
        final_key = (final_key << 8) | byte_val
    
    return final_key

def generate_key_candidates_from_rk1(sbox_inputs):
    """
    Generates all 256 possible 64-bit DES keys (with corrected parity) 
    that could result in the given 48-bit Round Key 1.
    
    Args:
        sbox_inputs: list of 8 integers (0-63), representing the 6-bit chunks of RK1.
                     
    Returns:
        list[int]: List of 256 possible 64-bit keys.
    """
    # 1. Flatten RK1
    rk1_bits = []
    for val in sbox_inputs:
        bits = [(val >> i) & 1 for i in range(5, -1, -1)]
        rk1_bits.extend(bits)
        
    # 2. Identify Missing Bits in C1D1 (Inverse PC2)
    # PC2 chooses 48 indices from 0..55. 
    # Create the base C1D1 and track which indices are UNKNOWN.
    c1d1_template = [None] * 56
    
    # Fill known bits
    # Output[i] = Input[_PC2[i]] -> Input[_PC2[i]] = Output[i]
    for i in range(48):
        original_idx = _PC2[i]
        c1d1_template[original_idx] = rk1_bits[i]
        
    # Find missing indices
    missing_indices = [i for i, x in enumerate(c1d1_template) if x is None]
    # Should be 56 - 48 = 8 indices
    
    candidates = []
    
    # 3. Iterate 2^8 = 256 possibilities
    import itertools
    for guess in itertools.product([0, 1], repeat=len(missing_indices)):
        # Construct specific C1D1
        current_c1d1 = list(c1d1_template) # Copy
        for i, bit in enumerate(guess):
            current_c1d1[missing_indices[i]] = bit
            
        # Continue with Inverse Shift and PC1 (Same as reconstruct function)
        
        # 3a. Inverse Shift (1 bit right)
        shift = _LEFT_ROTATIONS[0] # 1
        c1 = current_c1d1[:28]
        d1 = current_c1d1[28:]
        c0 = c1[-shift:] + c1[:-shift]
        d0 = d1[-shift:] + d1[:-shift]
        c0d0 = c0 + d0
        
        # 3b. Inverse PC1
        key_bits = [0] * 64
        for i in range(56):
            original_key_idx = _PC1[i]
            key_bits[original_key_idx] = c0d0[i]
            
        # 3c. To Int
        key_int = 0
        for b in key_bits:
            key_int = (key_int << 1) | b
            
        # 3d. Correct Parity
        final_key = 0
        for i in range(8):
            shift_val = (7 - i) * 8
            byte_val = (key_int >> shift_val) & 0xFF
            top_7 = byte_val & 0xFE
            w = bin(top_7).count('1')
            if w % 2 == 0: byte_val |= 1
            else: byte_val &= 0xFE
            final_key = (final_key << 8) | byte_val
            
        candidates.append(final_key)
        
    return candidates

def reconstruct_key_from_rk16(sbox_inputs):
    """
    Reconstructs the 64-bit DES key from the 48-bit Round Key 16 (Decryption Round 1).
    K16 is derived from C16D16 via PC2.
    Since C16D16 == C0D0 (State Identity), we skip the Inverse Shift step.
    
    Args:
        sbox_inputs: list of 8 integers (0-63), representing the 6-bit chunks of RK16.
    
    Returns:
        int: The reconstructed 64-bit key.
    """
    # 1. Flatten RK16
    rk16_bits = []
    for val in sbox_inputs:
        val = int(val)
        bits = [(val >> i) & 1 for i in range(5, -1, -1)]
        rk16_bits.extend(bits)
    
    # 2. Inverse PC2: Map 48 bits back to 56 bits (C16D16 -> C0D0 with holes)
    c0d0 = [0] * 56
    for i in range(48):
        original_idx = _PC2[i]
        c0d0[original_idx] = rk16_bits[i]
        
    # 3. Inverse Shift: SKIPPED (C16D16 == C0D0)
    
    # 4. Inverse PC1
    key_bits = [0] * 64
    for i in range(56):
        original_key_idx = _PC1[i]
        key_bits[original_key_idx] = c0d0[i]
        
    # 5. To Int
    key_int = 0
    for b in key_bits:
        key_int = (key_int << 1) | b
        
    # 6. Correct Parity
    final_key = 0
    for i in range(8):
        shift = (7 - i) * 8
        byte_val = (key_int >> shift) & 0xFF
        top_7 = byte_val & 0xFE
        w = bin(top_7).count('1')
        if w % 2 == 0: byte_val |= 1
        else: byte_val &= 0xFE
        final_key = (final_key << 8) | byte_val
    
    return final_key

def generate_subkey_candidates_optimized(sbox_outputs, er0_48bit):
    """
    Generates all possible 4^8 combinations of Round Key 1 chunks
    by XORing 6-bit S-box input candidates with the trace's Expanded R0 bits.
    
    Args:
        sbox_outputs: list of 8 integers (0-15), predicted 4-bit outputs.
        er0_48bit: 48-bit integer representing the Expanded R0 for the trace.
    
    Returns:
        list[list[int]]: List of combinations (each is a list of 8 6-bit chunks).
    """
    import itertools
    all_chunks_candidates = [] # list of lists (each sublist has 4 candidates)
    
    for sb in range(8):
        # 1. Get 4 possible 6-bit inputs (X) for this S-box output (Y)
        x_inputs = get_sbox_input_candidates(sb, sbox_outputs[sb])
        
        # 2. Extract the 6-bit chunk of ER0 for this S-box
        # S1: bits 42-47, S2: bits 36-41 ... S8: bits 0-5
        shift = 42 - (sb * 6)
        er0_chunk = (er0_48bit >> shift) & 0x3f
        
        # 3. K_chunk = X ^ ER0_chunk
        k_chunks = [x ^ er0_chunk for x in x_inputs]
        all_chunks_candidates.append(k_chunks)
        
    # Return 4^8 combinations of K1 chunks
    return list(itertools.product(*all_chunks_candidates))

def generate_subkey_combinations(sbox_outputs):
    """
    DEPRECATED: Use generate_subkey_candidates_optimized to factor in ER0.
    Returns combinations of S-box INPUTS (not keys).
    """
    import itertools
    all_chunks = []
    for sb in range(8):
        cands = get_sbox_input_candidates(sb, sbox_outputs[sb])
        all_chunks.append(cands)
    return list(itertools.product(*all_chunks))

def verify_recovered_key_with_pydes(recovered_key_int, predicted_chunks):
    """
    Verifies that the reconstructed key actually generates the predicted Round Key 1 chunks
    using the pyDes implementation.
    """
    try:
        # 1. Setup pyDes with the recovered key
        k_bytes = recovered_key_int.to_bytes(8, 'big')
        d = des(k_bytes)
        
        # 2. Get Round Key 1 (Kn[0] in pyDes)
        # Kn is list of lists of bits (48 bits)
        rk1_bits = d.Kn[0]
        
        # 3. Split into 8 chunks of 6 bits
        real_chunks = []
        for i in range(8):
            chunk_bits = rk1_bits[i*6 : (i+1)*6]
            # Convert bits to int
            val = 0
            for b in chunk_bits:
                val = (val << 1) | b
            real_chunks.append(val)
            
        # 4. Compare
        matches = [r == p for r, p in zip(real_chunks, predicted_chunks)]
        if all(matches):
            return True, "Match"
        else:
            return False, f"Mismatch. Real pyDes derived: {real_chunks} vs Predicted: {predicted_chunks}"
    except Exception as e:
        return False, f"Verification Error: {e}"

def derive_rsa_crt(p_hex, q_hex, e_list=[3, 65537]):
    """
    Derives RSA CRT components from P and Q (rsatool logic).
    Tries multiple public exponents (default 3 and 65537).
    """
    if not p_hex or not q_hex: return None
    from math import gcd
    try:
        p = int(p_hex, 16)
        q = int(q_hex, 16)
        if p <= 1 or q <= 1: return None
        if gcd(q, p) != 1: return None
        
        n = p * q
        phi = (p - 1) * (q - 1)
        
        # Try exponents in order
        valid_e = None
        for e in e_list:
            if gcd(e, phi) == 1:
                valid_e = e
                break
        
        if valid_e is None:
            return None
            
        d = pow(valid_e, -1, phi)
        dp = d % (p - 1)
        dq = d % (q - 1)
        qinv = pow(q, -1, p)
        
        return {
            'N': f"{n:0288X}", 
            'D': f"{d:0288X}",
            'P': f"{p:0144X}",
            'Q': f"{q:0144X}",
            'DP': f"{dp:0144X}",
            'DQ': f"{dq:0144X}",
            'QINV': f"{qinv:0128X}",
            'E': valid_e
        }
    except Exception:
        return None

def compute_batch_des(inputs, key_hex, mode='encrypt'):
    """
    Computes DES output for a batch of inputs using a given key.
    Used to generate 'Virtual Plaintext' for subsequent 3DES rounds.
    
    Args:
        inputs (list[int]): List of 64-bit integer inputs.
        key_hex (str): 16-char hex string (64-bit key).
        mode (str): 'encrypt' or 'decrypt'.
        
    Returns:
        list[int]: List of 64-bit integer outputs.
    """
    try:
        import src.pyDes as pyDes
        key_bytes = bytes.fromhex(key_hex)
        # Single DES
        k = pyDes.des(key_bytes, pyDes.ECB, pad=None)
        
        outputs = []
        for val in inputs:
            # Convert int to 8 bytes
            val_bytes = val.to_bytes(8, 'big')
            if mode == 'encrypt':
                out_bytes = k.encrypt(val_bytes)
            else:
                out_bytes = k.decrypt(val_bytes)
            # Convert back to int
            outputs.append(int.from_bytes(out_bytes, 'big'))
            
        return outputs
    except Exception as e:
        logger.error(f"Batch DES computation failed: {e}")
        return []

def derive_emv_session_keys(master_key_hex, atc_hex):
    """
    Derives EMV Session Keys (KENC, KMAC, KDEK) from a Master Key and ATC.
    Standard EMV Option A derivation (Generates 16-byte keys).
    """
    try:
        from src.pyDes import triple_des, ECB
        mk_bytes = bytes.fromhex(master_key_hex)
        if len(mk_bytes) == 8:
             mk_bytes = mk_bytes * 3
        elif len(mk_bytes) == 16:
             k1 = mk_bytes[:8]
             k2 = mk_bytes[8:16]
             mk_bytes = k1 + k2 + k1
             
        cipher = triple_des(mk_bytes, ECB, pad=None)
        
        atc_bytes = bytes.fromhex(atc_hex.zfill(4))
        
        def derive_full_key(constant_byte):
            # Derivation Data 1 (Left Half): ATC || F0 || constant || 00000000
            data_l = atc_bytes + bytes.fromhex(f"F0{constant_byte:02X}00000000")
            sk_l = cipher.encrypt(data_l).hex().upper()
            
            # Derivation Data 2 (Right Half): ATC || 0F || constant || 00000000
            data_r = atc_bytes + bytes.fromhex(f"0F{constant_byte:02X}00000000")
            sk_r = cipher.encrypt(data_r).hex().upper()
            
            return sk_l + sk_r

        return {
            'KENC': derive_full_key(1),
            'KMAC': derive_full_key(2),
            'KDEK': derive_full_key(3)
        }
    except Exception as e:
        logger.error(f"Session key derivation failed: {e}")
        return None

def verify_rsa_against_modulus(p_hex, q_hex, modulus_hex):
    """
    Verifies if P * Q == Modulus mathematically.
    Returns True/False and a status message.
    """
    try:
        p = int(p_hex, 16)
        q = int(q_hex, 16)
        n_recover = p * q
        n_real = int(modulus_hex, 16)
        
        if n_recover == n_real:
            return True, "Verified (P*Q == N)"
        else:
            return False, f"Mismatch: P*Q != N (Diff: {n_recover - n_real})"
    except Exception as e:
        return False, f"Error: {e}"

def standardize_rsa_crt(components, target_length=144):
    """
    Standardizes RSA CRT components to a specific hex length (default 144 chars / 576 bits).
    Ensures all keys (P, Q, DP, DQ, QINV) are zero-padded to this length.
    """
    standardized = {}
    for k, v in components.items():
        if k in ['P', 'Q', 'DP', 'DQ', 'QINV']:
            # Pad to target_length
            standardized[k] = v.zfill(target_length)
        elif k in ['N', 'D']:
            # N and D are typically double the prime length (1152 bits -> 288 chars)
            standardized[k] = v.zfill(target_length * 2)
        else:
            standardized[k] = v
    return standardized
