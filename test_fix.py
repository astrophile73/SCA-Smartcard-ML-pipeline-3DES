#!/usr/bin/env python3
"""
Test script to verify 3DES normalization fix.
Tests that _load_norm() correctly loads per-key-type statistics.
"""

import sys
import os

# Add pipeline-code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline-code"))

from src.inference_3des import _load_norm

# Test 1: Verify _load_norm() function signature accepts key_type
print("=" * 60)
print("TEST 1: _load_norm() function signature")
print("=" * 60)

# NOTE: model_dir is the PARENT directory returned by _resolve_3des_model_root()
# which contains the "3des" subdirectory
model_dir = "3des-pipeline/models"
try:
    # Test with key_type parameter
    result = _load_norm(model_dir, stage=1, key_type="kenc")
    print(f"✓ _load_norm(model_dir, stage=1, key_type='kenc') works")
    print(f"  Result type: {type(result)}")
    if result:
        mean, std = result
        print(f"  Mean shape: {mean.shape if hasattr(mean, 'shape') else len(mean)}")
        print(f"  Std shape: {std.shape if hasattr(std, 'shape') else len(std)}")
    else:
        print("  ⚠ Returned None (will fall back to root path)")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Test 2: Test all key types
print("=" * 60)
print("TEST 2: Load stats for all key types")
print("=" * 60)

for key_type in ["kenc", "kmac", "kdek"]:
    for stage in [1, 2]:
        try:
            result = _load_norm(model_dir, stage=stage, key_type=key_type)
            status = "✓ FOUND" if result else "✗ NOT FOUND (will use batch fallback)"
            print(f"  {key_type:6s} stage {stage}: {status}")
        except Exception as e:
            print(f"  {key_type:6s} stage {stage}: ✗ ERROR - {e}")

print()

# Test 3: Check specific file paths that should exist
print("=" * 60)
print("TEST 3: Check specific normalization file paths")
print("=" * 60)

for key_type in ["kenc", "kmac", "kdek"]:
    for stage in [1, 2]:
        # Paths constructed by _load_norm() with key_type parameter
        mean_path = os.path.join(model_dir, "3des", key_type, f"mean_s{stage}.npy")
        std_path = os.path.join(model_dir, "3des", key_type, f"std_s{stage}.npy")
        print(f"{key_type:6s} stage {stage}:")
        print(f"  Mean: {mean_path}")
        print(f"    Exists: {os.path.exists(mean_path)}")
        print(f"  Std:  {std_path}")
        print(f"    Exists: {os.path.exists(std_path)}")

print()
print("=" * 60)
print("TEST CONCLUSION")
print("=" * 60)
print("If all key_types show ✓ FOUND for both stages, the fix is working!")
print("The _load_norm() function now correctly loads per-key-type statistics.")
