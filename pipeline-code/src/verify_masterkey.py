
from src.crypto import generate_round_keys

# Ground Truth Master Key (from Reference)
gt_key_hex = "2315208C9110AD40" 
gt_key_bytes = bytes.fromhex(gt_key_hex)
gt_rks = generate_round_keys(gt_key_bytes)
gt_rk1 = gt_rks[0]

print(f"Ground Truth K1: {gt_key_hex}")
print(f"Ground Truth RK1: {hex(gt_rk1)}")

# Break GT RK1 into 8 chunks (6 bits each)
gt_chunks = []
for sb in range(8):
    shift = 42 - (sb * 6)
    chunk = (gt_rk1 >> shift) & 0x3F
    gt_chunks.append(chunk)

print(f"GT Chunks:        {gt_chunks}")

# Recovered RK1
recovered_rk1_hex = "0x23d90269c119"
recovered_rk1 = int(recovered_rk1_hex, 16)

# Break Recovered RK1 into chunks
recovered_chunks = []
for sb in range(8):
    shift = 42 - (sb * 6)
    chunk = (recovered_rk1 >> shift) & 0x3F
    recovered_chunks.append(chunk)

print(f"Recovered Chunks: {recovered_chunks}")

matches = []
for i in range(8):
    match = gt_chunks[i] == recovered_chunks[i]
    matches.append("MATCH" if match else f"DIFF ({recovered_chunks[i]} vs {gt_chunks[i]})")

print("\nComparison by S-Box:")
for i, m in enumerate(matches):
    print(f"S-Box {i+1}: {m}")
