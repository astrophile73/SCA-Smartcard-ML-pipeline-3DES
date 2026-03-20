"""
Mastercard 3DES Accuracy Test - Simplified

Directly running the main pipeline to test on Mastercard traces.
"""

import os
import sys
import subprocess

# Change to workspace directory
os.chdir(r"I:\freelance\SCA Smartcard ML Pipeline-3des")

# Set environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Run the main pipeline on Mastercard 3DES traces
cmd = [
    sys.executable,
    "pipeline-code/main.py",
    "--mode", "attack",
    "--input_dir", r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard",
    "--output_dir", r"Output/mastercard_test",
    "--processed_dir", r"Output/mastercard_processed",
    "--card_type", "mastercard",
    "--scan_type", "3des",  # 3DES only, no RSA
    "--model_root", r"pipeline-code/models",
]

print("Running Mastercard 3DES attack...")
print(f"Command: {' '.join(cmd)}\n")

try:
    result = subprocess.run(cmd, capture_output=False, text=True)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
