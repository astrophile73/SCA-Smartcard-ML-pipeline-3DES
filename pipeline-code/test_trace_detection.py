#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from src.ingest import TraceDataset

input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"

print("=" * 60)
print("Testing 3DES traces:")
ds_3des = TraceDataset(input_dir, trace_type="3des", card_type="universal")
print(f"  Files found: {len(ds_3des.files)}")
print(f"  Total traces: {ds_3des.total_traces}")
for f in ds_3des.files[:3]:
    print(f"    - {f.split(chr(92))[-1]}")

print("\n" + "=" * 60)
print("Testing RSA traces:")
try:
    ds_rsa = TraceDataset(input_dir, trace_type="rsa", card_type="universal")
    print(f"  Files found: {len(ds_rsa.files)}")
    print(f"  Total traces: {ds_rsa.total_traces}")
    for f in ds_rsa.files[:3]:
        print(f"    - {f.split(chr(92))[-1]}")
except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
