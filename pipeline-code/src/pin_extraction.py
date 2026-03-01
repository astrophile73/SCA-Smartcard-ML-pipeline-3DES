import pandas as pd
import numpy as np
import binascii
from Cryptodome.Cipher import DES3
from src.utils import setup_logger
from src.crypto import derive_rsa_crt

logger = setup_logger("pin_extraction")

def xor_block(b1_hex, b2_hex):
    # Helper to XOR two hex strings
    i1 = int(b1_hex, 16)
    i2 = int(b2_hex, 16)
    res = i1 ^ i2
    return f"{res:016X}"

def _parse_iso9564_pin(dec_hex, pan=None):
    """
    Parses a decrypted PIN block according to ISO 9564 (Formats 0, 1, 2).
    """
    # Format 0 logic (Requires PAN)
    if pan:
        # Prepare PAN Block for XOR
        # PAN (12 digits excluding check digit) -> 0000 P P P P P P P P P P P P
        # Get rightmost 12 digits of PAN excluding check digit
        pan_str = str(pan).strip()
        if len(pan_str) > 12:
            pan_part = pan_str[-13:-1] # Exclude last digit (Luhn check digit)
        else:
            pan_part = pan_str
        
        pan_block = f"0000{pan_part}".zfill(16)
        
        # Apply XOR
        dec_hex = xor_block(dec_hex, pan_block)

    # Now verify format
    fmt = dec_hex[0]
    
    if fmt in ['0', '1', '2']:
        length = int(dec_hex[1], 16) # Length is a hex digit
        if length > 12 or length < 4: 
            return f"RAW:{dec_hex}"
            
        pin = dec_hex[2:2+length]
        
        # Validate padding
        padding_start = 2 + length
        padding = dec_hex[padding_start:]
        
        # Format 0/2 usually pad with F, Format 1 with random.
        # Strict F check handles 0/2. For 1, we might relax it if needed.
        if fmt in ['0', '2']:
             if all(c == 'F' for c in padding):
                return pin
        else:
            # Format 1 allows random padding, just return PIN
            return pin
            
    return f"RAW:{dec_hex}"

def decrypt_pin_block_3des(pin_block_hex, key_hex, pan=None):
    """
    Decrypts a PIN block using a 3DES key.
    """
    try:
        # Prepare 3DES Key
        if len(key_hex) == 32: # Double-length key
            key_bytes = binascii.unhexlify(key_hex)
            key_bytes += key_bytes[:8] # K1=K3 for 3-key operation
        elif len(key_hex) == 48: # Triple-length key
            key_bytes = binascii.unhexlify(key_hex)
        else:
            logger.warning(f"Invalid 3DES key length: {len(key_hex)}")
            return None
            
        cipher = DES3.new(key_bytes, DES3.MODE_ECB)
        pin_block = binascii.unhexlify(pin_block_hex)
        decrypted = cipher.decrypt(pin_block)
        
        dec_hex = binascii.hexlify(decrypted).decode().upper()
        return _parse_iso9564_pin(dec_hex, pan)
            
    except Exception as e:
        logger.error(f"3DES Decryption error: {e}")
        return None

def extract_pin_from_trace(meta_path, key_hex):
    """
    (3DES VERSION) Scans metadata I/O logs for VERIFY command and extracts PIN using 3DES key.
    """
    logger.info(f"Extracting PIN using 3DES key: {key_hex[:10]}...")
    
    if not key_hex or len(key_hex) < 32:
        logger.warning("Invalid Key for 3DES PIN extraction.")
        return None

    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        logger.error(f"Could not read metadata: {e}")
        return None
        
    # This function now finds the PIN block and calls the decryption function
    pin_block = _find_pin_block_in_meta(df)
    
    if pin_block:
        pin = decrypt_pin_block_3des(pin_block, key_hex)
        if pin and not pin.startswith("RAW"):
            logger.info(f"SUCCESS: 3DES PIN Extraction Verified. PIN: {pin}")
            return pin
        else:
            logger.warning(f"3DES PIN Decryption failed or returned raw data: {pin}")
            return None
    else:
        logger.warning("No valid PIN block found in I/O traces for 3DES.")
        return None

def extract_pin_from_trace_rsa(meta_path, rsa_results):
    """
    (RSA VERSION) Scans metadata I/O logs for VERIFY command and extracts PIN using RSA key.
    """
    logger.info("Extracting PIN using RSA key components...")

    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        logger.error(f"Could not read metadata: {e}")
        return None
    
    pin_block_hex = _find_pin_block_in_meta(df)
    
    if not pin_block_hex:
        logger.warning("No valid PIN block found in I/O traces for RSA.")
        return None

    # We need to find the correct RSA key for this trace.
    # Let's assume we use the key from the first trace for all.
    p_hex = str(rsa_results['RSA_CRT_P'][0]).strip().replace(" ", "").replace("\n", "").replace("\r", "")
    q_hex = str(rsa_results['RSA_CRT_Q'][0]).strip().replace(" ", "").replace("\n", "").replace("\r", "")

    if not p_hex or not q_hex:
        logger.error("P or Q components missing from RSA results.")
        return None
        
    # Derive D and N, which are needed for decryption
    derived_keys = derive_rsa_crt(p_hex, q_hex)
    if not derived_keys:
        logger.error("Failed to derive full RSA key from P and Q.")
        return None
        
    d = int(derived_keys['D'], 16)
    n = int(derived_keys['N'], 16)
    c = int(pin_block_hex, 16)

    try:
        # Perform RSA decryption: m = c^d mod n
        # Check if ciphertext < modulus
        if c >= n:
             logger.warning("RSA Decryption Warning: Ciphertext >= Modulus.")
        
        m = pow(c, d, n)
        
        # The decrypted message `m` needs to be converted to bytes
        # The size of n in bytes determines the block size
        block_size = (n.bit_length() + 7) // 8
        decrypted_bytes = m.to_bytes(block_size, 'big')
        
        dec_hex = binascii.hexlify(decrypted_bytes).decode().upper()
        logger.info(f"Decrypted Hex (len {len(dec_hex)}): {dec_hex[:30]}...")

        # The actual PIN block data might be at the end of the decrypted bytes
        pin = _parse_iso9564_pin(dec_hex)

        if pin and not pin.startswith("RAW"):
            logger.info(f"SUCCESS: RSA PIN Extraction Verified. PIN: {pin}")
            return pin
        else:
            logger.warning(f"RSA PIN Decryption returned raw/invalid: {pin}")
            return None

    except Exception as e:
        logger.error(f"RSA Decryption error: {e}")
        return None


def _find_pin_block_in_meta(df):
    """
    Internal helper to find the first valid PIN block from a metadata DataFrame.
    Supports 8-byte (3DES) and 144-byte (RSA) blocks.
    """
    # PARSING LOGIC for VISA VERIFY (Robust)
    # Search for header: 00 20 00 88 (or generic 00 20 00)
    
    # Heuristic: Combine all string fields
    full_str = ""
    for idx, row in df.iterrows():
         if 'EncryptedPIN' in row and pd.notna(row['EncryptedPIN']) and len(str(row['EncryptedPIN'])) >= 16:
             return str(row['EncryptedPIN'])
         
         s = str(row.get('C7', '')) + str(row.get('IO', '')) + str(row.get('apdu', ''))
         full_str += s.replace(" ", "").replace("0x", "").upper()
         
    # Now scan full string
    start_pos = 0
    while True:
        # Search for Verify Command P3=88 (RSA 144 bytes) or 08 (DES)
        # Try finding standard verify header (00 20)
        idx_v = full_str.find("002000", start_pos)
        
        # Fallback: Internal Authenticate (00 88) or Generate AC (80 AE)
        if idx_v == -1:
            idx_v = full_str.find("008800", start_pos)
        if idx_v == -1:
            idx_v = full_str.find("80AE", start_pos)
            if idx_v != -1:
                # 80 AE P1 P2 Lc
                # Payload usually starts at idx_v + 10 if Lc is at idx_v + 8
                pass 
                
        if idx_v == -1: 
            break
            
        try:
             # Header: CLA INS P1 P2 Lc
             # Length Lc is at idx_v + 8: idx_v + 10
             lc_hex = full_str[idx_v+8:idx_v+10]
             lc_val = int(lc_hex, 16)
             
             logger.debug(f"Found Command Header ({full_str[idx_v:idx_v+4]}) at {idx_v}, Lc={lc_hex} ({lc_val})")
             
             # Extract payload
             payload_start = idx_v + 10
             payload_end = payload_start + (lc_val * 2)
             
             if payload_end <= len(full_str):
                  payload = full_str[payload_start:payload_end]
                  # Filter sanity checks for PIN blocks
                  if lc_val in [8, 16, 128, 144]: # 8=DES, 16=Double DES PIN, 128/144=RSA
                      return payload
             else:
                  logger.warning(f"Truncated payload at {idx_v}")
                 
        except Exception as e:
            pass
            
        start_pos = idx_v + 6
        
    return None

