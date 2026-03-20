#!/usr/bin/env python3
import sys
sys.path.insert(0, 'pipeline-code')
from src.inference_3des import recover_3des_keys

predicted = recover_3des_keys(
    processed_dir='Output/mastercard_processed',
    model_dir='pipeline-code/models',
    card_type='mastercard',
    n_attack=10000
)

gt_kenc = '9E15204313F7318ACB79B90BD986AD29'
gt_kmac = '4664942FE615FB02E5D57F292AA2B3B6'
gt_kdek = 'CE293B8CC12A977379EF256D76109492'

pred_kenc = predicted.get('3DES_KENC', '')
pred_kmac = predicted.get('3DES_KMAC', '')
pred_kdek = predicted.get('3DES_KDEK', '')

print('\n' + '='*80)
print('PREDICTIONS vs GROUND TRUTH')
print('='*80)
print(f'\nPREDICTED:')
print(f'  KENC: {pred_kenc}')
print(f'  KMAC: {pred_kmac}')
print(f'  KDEK: {pred_kdek}')

print(f'\nGROUND TRUTH:')
print(f'  KENC: {gt_kenc}')
print(f'  KMAC: {gt_kmac}')
print(f'  KDEK: {gt_kdek}')

print(f'\nDIRECT MATCH:')
print(f'  KENC: {pred_kenc.upper() == gt_kenc.upper()}')
print(f'  KMAC: {pred_kmac.upper() == gt_kmac.upper()}')
print(f'  KDEK: {pred_kdek.upper() == gt_kdek.upper()}')

# Check permutations
from itertools import permutations
print(f'\nKEY SLOT PERMUTATION TEST:')
key_order = ['KENC', 'KMAC', 'KDEK']
pred_vals = [pred_kenc, pred_kmac, pred_kdek]
gt_vals = [gt_kenc, gt_kmac, gt_kdek]

best_match = 0
best_perm_str = ""
for perm in permutations(range(3)):
    matches = 0
    if pred_vals[perm[0]].upper() == gt_kenc.upper():
        matches += 1
    if pred_vals[perm[1]].upper() == gt_kmac.upper():
        matches += 1
    if pred_vals[perm[2]].upper() == gt_kdek.upper():
        matches += 1
    
    mapping = f"{key_order[perm[0]]}→KENC, {key_order[perm[1]]}→KMAC, {key_order[perm[2]]}→KDEK"
    print(f"  {mapping}: {matches}/3 matches")
    
    if matches > best_match:
        best_match = matches
        best_perm_str = mapping

if best_match > 0:
    print(f'\nBEST PERMUTATION: {best_perm_str} ({best_match}/3 matches)')
else:
    print(f'\nNO PERMUTATION MATCHES ANY KEY')

print('='*80)
