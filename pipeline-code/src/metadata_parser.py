import numpy as np
import os

class MetadataParser:
    """
    Parses C7 I/O traces to extract EMV metadata (PAN, ATC, Track 2) 
    and identifies the GENERATE AC trigger for SCA alignment.
    """
    
    TAGS = {
        '5A': 'PAN',
        '5F24': 'Expiry Date',
        '9F36': 'ATC',
        '57': 'TRACK2', # Track 2 Equivalent Data
        '9F37': 'Unpredictable Number',
        '82': 'AIP',
        '9F10': 'IAD'
    }

    def __init__(self, trace_path):
        self.trace_path = trace_path
        self.acr_send = []
        self.acr_receive = []
        self.track2_raw = ""
        self.metadata = {}

    def load_trace(self):
        """Loads the .npz file or directory and extracts protocol strings."""
        try:
            if os.path.isdir(self.trace_path):
                # Loose files in directory
                class LooseData:
                    def __init__(self, d): 
                        self.d = d
                    def __getitem__(self, k): 
                        p = os.path.join(self.d, k + ".npy")
                        return np.load(p, allow_pickle=True) if os.path.exists(p) else None
                    def __contains__(self, k): 
                        return os.path.exists(os.path.join(self.d, k + ".npy"))
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                data_manager = LooseData(self.trace_path)
            else:
                data_manager = np.load(self.trace_path, mmap_mode='r', allow_pickle=True)

            with data_manager as data:
                # Handle scalar vs array for ACR_send
                send_keys = ['ACR_send', 'Verify_command', 'apdu']
                self.acr_send = []
                for k in send_keys:
                    if k in data and data[k] is not None:
                        val = data[k]
                        if isinstance(val, np.ndarray) and val.ndim == 0:
                            self.acr_send.append(str(val.item()))
                        elif isinstance(val, (np.ndarray, list)):
                            # If it's a byte trace, convert to hex string
                            if isinstance(val, np.ndarray) and val.dtype.kind in ['u', 'i'] and val.ndim > 0:
                                import binascii
                                self.acr_send.append(binascii.hexlify(val).decode().upper())
                            else:
                                self.acr_send.extend([str(x) for x in val if x is not None])
                        else:
                            self.acr_send.append(str(val))

                # Handle scalar vs array for ACR_receive
                recv_keys = ['ACR_receive', 'Verify_response']
                self.acr_receive = []
                for k in recv_keys:
                    if k in data and data[k] is not None:
                        val = data[k]
                        if isinstance(val, np.ndarray) and val.ndim == 0:
                            self.acr_receive.append(str(val.item()))
                        elif isinstance(val, (np.ndarray, list)):
                            self.acr_receive.extend([str(x) for x in val if x is not None])
                        else:
                            self.acr_receive.append(str(val))

                if 'Track2' in data and data['Track2'] is not None:
                    t2 = data['Track2']
                    if isinstance(t2, np.ndarray):
                        if t2.size == 1:
                            t2 = t2.item()
                        else:
                            t2 = t2[0] if t2.ndim > 0 else t2
                    
                    if isinstance(t2, bytes):
                        self.track2_raw = t2.decode('utf-8')
                    else:
                        self.track2_raw = str(t2)
            
            # Post-process: If acr_receive is a single string with newlines/spaces, split it
            # This handles CSVs where multiple APDUs are packed into one cell
            new_receive = []
            for item in self.acr_receive:
                if isinstance(item, str) and ('\n' in item or len(item) > 500): 
                    # Heuristic: split by \n or naive length check isn't enough, 
                    # but usually these logs are \n separated.
                    parts = item.replace('\r', '\n').split('\n')
                    new_receive.extend([p.strip() for p in parts if p.strip()])
                else:
                    new_receive.append(item)
            if len(new_receive) > len(self.acr_receive):
                self.acr_receive = new_receive
                
            return True
        except Exception as e:
            print(f"Error loading trace metadata: {e}")
            return False

    def parse_tlv(self, hex_str):
        """Rudimentary BER-TLV parser for hex strings."""
        if not hex_str: return {}
        results = {}
        i = 0
        while i < len(hex_str):
            try:
                tag = hex_str[i:i+2]
                if tag == '00' or tag == 'FF': # Padding/End
                    i += 2
                    continue

                if (int(tag, 16) & 0x1F) == 0x1F: # Two-byte tag
                    tag = hex_str[i:i+4]
                    i += 4
                else:
                    i += 2
                
                if i >= len(hex_str): break
                length_byte = int(hex_str[i:i+2], 16)
                if length_byte & 0x80: # Long form length
                    num_bytes = length_byte & 0x7F
                    length = int(hex_str[i+2:i+2+(num_bytes*2)], 16)
                    i += 2 + (num_bytes*2)
                else:
                    length = length_byte
                    i += 2
                
                value = hex_str[i:i+(length*2)]
                results[tag.upper()] = value
                
                results[tag.upper()] = value
                
                # Recursive call if tag is constructed (e.g., 77, 70, 61, 6F)
                if int(tag[:2], 16) & 0x20:
                    results.update(self.parse_tlv(value))
                
                i += (length*2)
            except:
                break
        return results

    def extract_metadata(self):
        """Orchestrates extraction of PAN, ATC, and Expiry."""
        # Clean up track2_raw in case of b'' wrapper
        t2 = self.track2_raw
        if t2.startswith("b'") and t2.endswith("'"):
            t2 = t2[2:-1]
        
        # 1. Parse Track 2 if available
        if t2:
            if 'D' in t2.upper():
                parts = t2.upper().split('D')
                self.metadata['PAN'] = parts[0]
                self.metadata['Expiry'] = parts[1][:4]

        # 2. Robust Recursive Scan of ALL Responses
        # We ignore the command (send_msg) solely for finding these tags, 
        # as the command logs might be non-standard (e.g. 0088...)
        for recv_msg in self.acr_receive:
            if not recv_msg: continue
            clean_msg = str(recv_msg).replace(" ", "").upper()
            
            # recursive scan
            tlv = self.parse_tlv(clean_msg)
            
            # Check for critical tags in the parsed dictionary
            # AIP (Tag 82)
            if '82' in tlv and 'AIP' not in self.metadata:
                self.metadata['AIP'] = tlv['82']
            
            # IAD (Tag 9F10)
            if '9F10' in tlv and 'IAD' not in self.metadata:
                self.metadata['IAD'] = tlv['9F10']
                
            # ATC (Tag 9F36)
            if '9F36' in tlv and 'ATC' not in self.metadata:
                self.metadata['ATC'] = tlv['9F36']
                
            # Fallback: Check Template 77/80 contents explicitly if not found
            if '77' in tlv:
                inner = self.parse_tlv(tlv['77'])
                if '82' in inner and 'AIP' not in self.metadata: self.metadata['AIP'] = inner['82']
                if '9F10' in inner and 'IAD' not in self.metadata: self.metadata['IAD'] = inner['9F10']
                if '9F36' in inner and 'ATC' not in self.metadata: self.metadata['ATC'] = inner['9F36']
            
            if '80' in tlv:
                # 80 is usually primitive, but sometimes contains concatenated data
                # We can't parse it easily without context, but if we are desperate:
                val = tlv['80']
                # Heuristic: If we don't have ATC, and we see an 80 tag response that is reasonably long...
                pass
        
        # 4. Derive PAN from TRACK2 if PAN is missing
        if not self.metadata.get('PAN') and self.metadata.get('TRACK2'):
            t2 = self.metadata['TRACK2']
            if 'D' in t2.upper():
                self.metadata['PAN'] = t2.upper().split('D')[0]
            else:
                self.metadata['PAN'] = t2 

        # 5. Extract Encrypted PIN from VERIFY command (00 20) in ACR_send
        # Command Structure: CLA INS P1 P2 Lc [Data]
        # Verify: 00 20 00 80 08 [Encrypted PIN Block]
        for log_entry in self.acr_send:
            if not log_entry: continue
            # Split log entry by newlines (as multiple APDUs are in one entry)
            lines = str(log_entry).replace('\r', '\n').split('\n')
            
            for line in lines:
                clean_cmd = line.replace(" ", "").upper()
                # Remove timestamps or prefixes if any (usually log starts with command bytes or valid hex)
                # Heuristic: Find '0020'
                if '0020' in clean_cmd:
                    # Find index where 0020 starts
                    idx = clean_cmd.find('0020')
                    # Treat everything after as the command
                    cmd_part = clean_cmd[idx:]
                    
                    # Check P2 (00 20 xx 80/88)
                    if len(cmd_part) >= 12: 
                             # Simple parse: skip 8 chars (CLA INS P1 P2), read Lc (2 chars)
                             try:
                                 lc = int(cmd_part[8:10], 16)
                                 if len(cmd_part) >= 10 + (lc * 2):
                                     pin_block = cmd_part[10 : 10 + (lc*2)]
                                     # Supports 8 bytes (16 hex chars) for 3DES or 144 bytes (288 hex chars) for RSA
                                     if len(pin_block) >= 16:
                                         self.metadata['EncryptedPIN'] = pin_block
                                         return self.metadata
                             except:
                                 pass 
        
        return self.metadata

    def get_generate_ac_trigger(self):
        """Finds the index in the trace where INS=0xAE (GENERATE AC) occurs."""
        # This will be used to align the C1 Power traces
        pass

if __name__ == "__main__":
    # Test initialization
    print("Metadata Parser initialized for Phase 2.")
