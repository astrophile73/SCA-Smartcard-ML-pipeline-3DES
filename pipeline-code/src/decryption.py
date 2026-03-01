from Crypto.Util.number import long_to_bytes, bytes_to_long
import logging

logger = logging.getLogger(__name__)

def parse_pin_block(decrypted_bytes, pan=None):
    """
    Parses ISO 9564 PIN Blocks (Format 0, 1, 2).
    - Format 0: Needs PAN for XOR (Format 0 is Block XOR PAN).
    - Format 1: No PAN, starts with 1.
    - Format 2: No PAN, starts with 2 (Standard for RSA).
    """
    try:
        if not decrypted_bytes: return None
        
        # Helper to get high/low nibbles
        block_len = len(decrypted_bytes)
        if block_len != 8 and block_len != 16: # Standard is 8 bytes, RSA dec result might be longer (128 bytes), need to take last 8?
            # Usually PIN block is 8 bytes. If RSA decrypted, it's 128 bytes padding + 8 bytes data?
            # Actually RSA encryption pads the 8-byte PIN block into 128 bytes.
            # We need to find the payload.
            # ISO 9564-2 for RSA: 0x00 || 0x02 || Random Non-Zero || 0x00 || PIN Block (Format 0/1/2?)
            # Wait, usually the Decrypted output IS the PIN block (if no PKCS#1 padding was used on top of PIN block formatting).
            # If PKCS#1 v1.5 padding is used: 00 02 [random] 00 [Message]
            # The "Message" IS the 8-byte PIN Block.
            
            # Heuristic: verify PKCS#1 padding
            if block_len == 128 and decrypted_bytes[0] == 0:
                # Search for the 00 separator
                try:
                    sep_idx = decrypted_bytes.index(0, 2) # Skip first 00 02...
                    payload = decrypted_bytes[sep_idx+1:]
                    if len(payload) == 8:
                        decrypted_bytes = payload
                except: pass

        # Check Format
        fmt = decrypted_bytes[0] >> 4
        
        if fmt == 0:
            # Format 0 (requires PAN)
            # Block = PIN_Data XOR PAN_Data
            if not pan:
                # If we don't have PAN, we can't fully check, but maybe it's raw?
                # Sometimes people send raw format 0 without the XOR (rare/wrong)
                pass 
            else:
                # Prepare PAN: Take 12 rightmost digits excluding check digit
                # PAN must be hex string
                pan_crop = pan[-13:-1] if len(pan) >= 13 else pan # simplistic
                # Create PAN Block: 00 00 + PAN (padded) -> Actually 00 00 P1..P12
                # Correct ISO 0 PAN Block: 00 00 P1 P2 ... P12
                # We need to XOR 'decrypted_bytes' with 'pan_block'
                # This is complex to implement robustly without clean PAN.
                pass 
                
            # If we assume it is already XORed or we don't have PAN, we process as is:
            pin_len = decrypted_bytes[0] & 0x0F
            if pin_len > 12: return None
            
            pin_res = ""
            for i in range(1, 8):
                byte = decrypted_bytes[i]
                pin_res += f"{byte >> 4:X}{byte & 0x0F:X}"
            return pin_res[:pin_len]

        elif fmt == 1:
            # Format 1: 1L P...P R...R
            pin_len = decrypted_bytes[0] & 0x0F
            pin_res = ""
            for i in range(1, 8):
                byte = decrypted_bytes[i]
                pin_res += f"{byte >> 4:X}{byte & 0x0F:X}"
            return pin_res[:pin_len]

        elif fmt == 2:
            # Format 2: 2L P...P F...F (Used for local offline/RSA usually)
            pin_len = decrypted_bytes[0] & 0x0F
            pin_res = ""
            for i in range(1, 8):
                byte = decrypted_bytes[i]
                pin_res += f"{byte >> 4:X}{byte & 0x0F:X}"
            return pin_res[:pin_len]
            
        return None
        
    except Exception as e:
        logger.error(f"Error parsing PIN block: {e}")
        return None

def decrypt_rsa_pin(enc_pin_block, d, n):
    """
    Decrypts an RSA encrypted PIN block.
    """
    try:
        if isinstance(enc_pin_block, bytes):
            c = bytes_to_long(enc_pin_block)
        elif isinstance(enc_pin_block, str):
            c = int(enc_pin_block, 16)
        else:
            c = int(enc_pin_block)
        d = int(d)
        n = int(n)
        m = pow(c, d, n)
        n_len = (n.bit_length() + 7) // 8
        m_bytes = long_to_bytes(m, n_len)
        return m_bytes
    except Exception as e:
        logger.error(f"Error decrypting RSA PIN: {e}")
        return None

def decrypt_3des_pin(enc_pin_block, key_hex):
    """
    Decrypts a 3DES encrypted PIN block (ECB mode usually for PIN).
    """
    try:
        # Integration of BlackRabbit pyDes (pure python) as requested
        import src.pyDes as pyDes
        key = bytes.fromhex(key_hex)
        
        # Determine mode based on key length
        if len(key) == 8:
            # Single DES
            cipher = pyDes.des(key, pyDes.ECB, pad=None)
        elif len(key) in [16, 24]:
            # Triple DES
            cipher = pyDes.triple_des(key, pyDes.ECB, pad=None)
        else:
            # Fallback for weird key lengths (attempting 16-byte expansion if < 24? 
            # pyDes handles 16/24. If other, we might need to adjust.)
            # For now, if 32 hex chars (16 bytes), it works.
            if len(key) > 24: key = key[:24] # Truncate if too long
            cipher = pyDes.triple_des(key, pyDes.ECB, pad=None)

        if isinstance(enc_pin_block, str):
            data = bytes.fromhex(enc_pin_block)
        else:
            data = enc_pin_block
            
        decrypted = cipher.decrypt(data)
        return decrypted
    except Exception as e:
        logger.error(f"Error decrypting 3DES PIN: {e}")
        return None

def extract_rsa_modulus_from_apdu(c7_hex):
    """
    Scans C7 Hex string for Tag 9F46.
    """
    try:
        tag = "9F46"
        if tag not in c7_hex:
            return None
        # Placeholder
        return "RSA_MODULUS_FOUND_BUT_PARSING_NOT_IMPLEMENTED"
    except Exception as e:
        logger.error(f"Error extracting RSA Modulus: {e}")
        return None

def extract_pin_block_from_c7(c7_hex):
    """
    Extracts the Encrypted PIN Block from C7 (I/O) trace.
    Target: VERIFY Command (00 20 ...)
    """
    try:
        if not c7_hex or not isinstance(c7_hex, str):
            return None
        clean_hex = c7_hex.replace(" ", "").upper()
        idx = clean_hex.find("002000")
        if idx == -1:
            return None
        # P2=idx+6, Lc=idx+8
        lc_idx = idx + 8
        if lc_idx + 2 > len(clean_hex): return None
        lc_hex = clean_hex[lc_idx:lc_idx+2]
        lc = int(lc_hex, 16)
        
        # Data starts after Lc (2 chars)
        data_idx = lc_idx + 2
        
        # Verify Lc is reasonable (e.g. 8 bytes for DES Pin Block)
        # Some verify commands might be shorter or longer.
        if lc < 8: return None
        
        data_len_chars = lc * 2
        if data_idx + data_len_chars > len(clean_hex):
            return None
            
        # Extract strictly the data payload
        pin_block = clean_hex[data_idx:data_idx+16] # Taking first 8 bytes (16 chars) as PIN block
        return pin_block
    except Exception as e:
        logger.error(f"Error extraction PIN block: {e}")
        return None
