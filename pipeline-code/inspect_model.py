#!/usr/bin/env python
"""
Inspect a trained 3DES model to check if it contains actual weights or is just initialization.
"""
import torch
import os

model_path = r"i:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\models\3des\kenc\s1\sbox1_model0.pth"

print("="*80)
print(f"Inspecting model: {os.path.basename(model_path)}")
print("="*80)

try:
    state_dict = torch.load(model_path, map_location='cpu')
    
    print(f"\nModel state_dict keys: {len(state_dict)}")
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if tensor.dtype.is_floating_point:
            print(f"  {key:30s} shape={str(tensor.shape):15s} dtype={tensor.dtype} "
                  f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
                  f"mean={tensor.mean().item():.6f}")
        else:
            print(f"  {key:30s} shape={str(tensor.shape):15s} dtype={tensor.dtype} "
                  f"(non-float, skipped)")
    
    # Check if weights look trained or random/zero
    print(f"\n" + "="*80)
    print("Weight Analysis")
    print("="*80)
    
    for key, tensor in state_dict.items():
        if 'weight' in key.lower() and tensor.dtype.is_floating_point:
            unique_vals = torch.unique(tensor).numel()
            zero_ratio = (tensor == 0).float().mean().item()
            print(f"\n{key}:")
            print(f"  Unique values: {unique_vals}")
            print(f"  Zero ratio: {zero_ratio:.4f}")
            print(f"  Std dev: {tensor.std().item():.6f}")
            
            # Check if weights look like random initialization or trained
            if unique_vals == 1 and abs(tensor.mean().item()) < 1e-6:
                print(f"  → SUSPICIOUS: All weights are ZERO (untrained?)")
            elif unique_vals == 2:
                print(f"  → SUSPICIOUS: Only 2 unique values (binary?)")
            elif unique_vals < 100:
                print(f"  → SUSPICIOUS: Very few unique values ({unique_vals})")
            elif zero_ratio > 0.9:
                print(f"  → SUSPICIOUS: Mostly sparse (90% zeros)")
            else:
                print(f"  → Looks reasonable for trained model")
    
    print(f"\n" + "="*80)
    print("Compare with untrained model initialization...")
    print("="*80)
    
    # Create a fresh untrained model and compare
    import sys
    sys.path.insert(0, r"i:\freelance\SCA Smartcard ML Pipeline-3des\pipeline-code")
    from src.model import get_model
    
    untrained = get_model(input_dim=200, num_classes=16, use_shared_backbone=False)
    untrained_state = untrained.state_dict()
    
    print(f"\nComparing layer fc1.weight:")
    if 'fc1.weight' in state_dict and 'fc1.weight' in untrained_state:
        trained_weight = state_dict['fc1.weight']
        untrained_weight = untrained_state['fc1.weight']
        
        # Check correlation
        flat_trained = trained_weight.view(-1)
        flat_untrained = untrained_weight.view(-1)
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_trained.unsqueeze(0), 
            flat_untrained.unsqueeze(0)
        ).item()
        
        print(f"  Trained model fc1 mean: {trained_weight.mean().item():.6f}")
        print(f"  Untrained model fc1 mean: {untrained_weight.mean().item():.6f}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        
        if abs(cos_sim - 1.0) < 0.1:  # High similarity = similar initialization
            print(f"  → SUSPICIOUS: Very similar to untrained initialization!")
        elif abs(cos_sim + 1.0) < 0.1:  # Negative similarity = opposite
            print(f"  → May be trained (very different)")
        else:
            print(f"  → Looks somewhat different from initialization")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
