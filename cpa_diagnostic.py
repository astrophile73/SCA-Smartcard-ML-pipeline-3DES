"""
CPA Diagnostic: Analyze trace characteristics
"""

import pandas as pd
import numpy as np
from pathlib import Path

input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
csv_file = input_dir + "\\traces_data_1000T_1.csv"

print("Loading first 100 traces...")
df = pd.read_csv(csv_file, nrows=100)

print(f"\nDataframe shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Parse a few traces
traces_list = []
for i in range(5):
    trace_str = str(df.iloc[i]['trace_data'])
    samples = [float(x.strip()) for x in trace_str.split(',')]
    traces_list.append(samples)
    print(f"\nTrace {i}:")
    print(f"  Length: {len(samples)}")
    print(f"  Min: {min(samples):.6f}, Max: {max(samples):.6f}, Mean: {np.mean(samples):.6f}")
    print(f"  First 10 samples: {samples[:10]}")

# Check if all traces have same length
all_traces = []
for idx in range(len(df)):
    trace_str = str(df.iloc[idx]['trace_data'])
    samples = [float(x.strip()) for x in trace_str.split(',')]
    all_traces.append(samples)

all_traces = np.array(all_traces)
print(f"\nAll traces shape: {all_traces.shape}")
print(f"Overall statistics:")
print(f"  Min: {all_traces.min():.6f}")
print(f"  Max: {all_traces.max():.6f}")
print(f"  Mean: {all_traces.mean():.6f}")
print(f"  Std: {all_traces.std():.6f}")

# Check power variation across traces at each time point
print(f"\nPower variation at specific time points:")
for t in [0, 100, 200, 500]:
    if t < all_traces.shape[1]:
        vals = all_traces[:, t]
        print(f"  Time {t}: min={vals.min():.4f}, max={vals.max():.4f}, std={vals.std():.6f}")

# Check power distribution per trace (sum)
power_sums = np.sum(np.abs(all_traces), axis=1)
print(f"\nPower sums (sum of absolute values):")
print(f"  Min: {power_sums.min():.2f}, Max: {power_sums.max():.2f}, Mean: {power_sums.mean():.2f}, Std: {power_sums.std():.2f}")
