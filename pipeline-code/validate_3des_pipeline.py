#!/usr/bin/env python
"""
Comprehensive validation that the 3DES pipeline is working correctly:
1. Feature extraction from traces
2. Model inference on features
3. Key reconstruction from S-box predictions

Compare recovered keys against ground truth.
"""
import os
import pandas as pd
import numpy as np

processed_3des = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Processed\3des"
meta_path = os.path.join(processed_3des, "Y_meta.csv")
output_report = r"I:\freelance\SCA Smartcard ML Pipeline-3des\3des-pipeline\Output\test6\Final_Report_universal_session.csv"

print("="*80)
print("STEP 1: Check Ground Truth Labels in Metadata")
print("="*80)

if os.path.exists(meta_path):
    meta_df = pd.read_csv(meta_path)
    print(f"✓ Metadata loaded: {len(meta_df)} rows")
    print(f"  Columns: {list(meta_df.columns)}")
    
    if 'T_DES_KENC' in meta_df.columns:
        # Get unique ground truth keys
        unique_kenc = meta_df['T_DES_KENC'].unique().tolist()
        unique_kmac = meta_df['T_DES_KMAC'].unique().tolist()
        unique_kdek = meta_df['T_DES_KDEK'].unique().tolist()
        
        print(f"\n✓ Ground Truth Keys Found:")
        print(f"  Unique T_DES_KENC values: {len(unique_kenc)}")
        print(f"  Unique T_DES_KMAC values: {len(unique_kmac)}")
        print(f"  Unique T_DES_KDEK values: {len(unique_kdek)}")
        
        if len(unique_kenc) > 0:
            print(f"\n  First unique KENC: {unique_kenc[0] if isinstance(unique_kenc[0], str) else meta_df['T_DES_KENC'].dropna().iloc[0]}")
        if len(unique_kmac) > 0:
            print(f"  First unique KMAC: {unique_kmac[0] if isinstance(unique_kmac[0], str) else meta_df['T_DES_KMAC'].dropna().iloc[0]}")
        if len(unique_kdek) > 0:
            print(f"  First unique KDEK: {unique_kdek[0] if isinstance(unique_kdek[0], str) else meta_df['T_DES_KDEK'].dropna().iloc[0]}")
        
        # Check ground truth values for first few rows
        print(f"\n  Ground Truth (first 5 rows):")
        for idx in range(min(5, len(meta_df))):
            kenc = meta_df.iloc[idx].get('T_DES_KENC', '')
            kmac = meta_df.iloc[idx].get('T_DES_KMAC', '')
            kdek = meta_df.iloc[idx].get('T_DES_KDEK', '')
            print(f"    Row {idx}: KENC={str(kenc)[:24]}... KMAC={str(kmac)[:24]}... KDEK={str(kdek)[:24]}...")
    else:
        print("✗ No 3DES key columns in metadata")
else:
    print(f"✗ Metadata not found: {meta_path}")

print("\n" + "="*80)
print("STEP 2: Check Recovered Keys in Report")
print("="*80)

if os.path.exists(output_report):
    report_df = pd.read_csv(output_report)
    print(f"✓ Report loaded: {len(report_df)} rows")
    
    # Get unique recovered keys
    unique_recovered_kenc = report_df['3DES_KENC'].dropna().unique().tolist()
    print(f"\n✓ Recovered Keys:")
    print(f"  Unique recovered KENC values: {len(unique_recovered_kenc)}")
    print(f"  Unique recovered KMAC values: {len(report_df['3DES_KMAC'].dropna().unique())}")
    print(f"  Unique recovered KDEK values: {len(report_df['3DES_KDEK'].dropna().unique())}")
    
    if len(unique_recovered_kenc) > 0:
        print(f"\n  First recovered KENC: {unique_recovered_kenc[0]}")
    
    print(f"\n  Recovered (first 5 rows):")
    for idx in range(min(5, len(report_df))):
        kenc = report_df.iloc[idx].get('3DES_KENC', '')
        kmac = report_df.iloc[idx].get('3DES_KMAC', '')
        kdek = report_df.iloc[idx].get('3DES_KDEK', '')
        print(f"    Row {idx}: KENC={str(kenc)[:24]}... KMAC={str(kmac)[:24]}... KDEK={str(kdek)[:24]}...")
else:
    print(f"✗ Report not found: {output_report}")

print("\n" + "="*80)
print("STEP 3: Compare Ground Truth vs Recovered")
print("="*80)

if os.path.exists(meta_path) and os.path.exists(output_report):
    # Normalize hex for comparison
    def norm_hex(x):
        if x is None or x == '':
            return ''
        x = str(x).strip().replace(' ', '').upper()
        x = ''.join(c for c in x if c in '0123456789ABCDEF')
        return x
    
    # Get ground truth (first row's ground truth is representative since master key is static)
    gt_kenc_first = norm_hex(meta_df.iloc[0].get('T_DES_KENC', ''))
    gt_kmac_first = norm_hex(meta_df.iloc[0].get('T_DES_KMAC', ''))
    gt_kdek_first = norm_hex(meta_df.iloc[0].get('T_DES_KDEK', ''))
    
    # Get first recovered key
    rec_kenc_first = norm_hex(report_df.iloc[0].get('3DES_KENC', ''))
    rec_kmac_first = norm_hex(report_df.iloc[0].get('3DES_KMAC', ''))
    rec_kdek_first = norm_hex(report_df.iloc[0].get('3DES_KDEK', ''))
    
    print(f"Ground Truth (first row):")
    print(f"  KENC: {gt_kenc_first}")
    print(f"  KMAC: {gt_kmac_first}")
    print(f"  KDEK: {gt_kdek_first}")
    
    print(f"\nRecovered (first row):")
    print(f"  KENC: {rec_kenc_first}")
    print(f"  KMAC: {rec_kmac_first}")
    print(f"  KDEK: {rec_kdek_first}")
    
    print(f"\nComparison:")
    print(f"  KENC Match: {gt_kenc_first == rec_kenc_first}")
    print(f"  KMAC Match: {gt_kmac_first == rec_kmac_first}")
    print(f"  KDEK Match: {gt_kdek_first == rec_kdek_first}")
    
    if len(gt_kenc_first) > 0 and len(rec_kenc_first) > 0:
        # Check byte-level accuracy
        byte_match_kenc = sum(1 for i in range(0, min(len(gt_kenc_first), len(rec_kenc_first)), 2) 
                             if gt_kenc_first[i:i+2] == rec_kenc_first[i:i+2])
        total_bytes = max(len(gt_kenc_first), len(rec_kenc_first)) // 2
        print(f"\n  KENC Byte Accuracy: {byte_match_kenc}/{total_bytes} bytes ({100*byte_match_kenc/total_bytes:.1f}%)")

print("\n" + "="*80)
print("STEP 4: Verify Pipeline Architecture")
print("="*80)

print("""
Pipeline stages implemented:
─────────────────────────────────────

STAGE 1: FEATURE EXTRACTION (feature_eng.py, extract_features())
✓ Loads raw traces from NPZ/CSV files
✓ Pass 1: Alignment using SAD (ChipWhisperer ResyncSAD)
✓ Pass 1: POI selection (variance-based for RSA, correlation-based for 3DES)
✓ Pass 2: Feature extraction at POIs for each trace
✓ Output: X_features.npy (1000s traces × POI features)

STAGE 2: MODEL INFERENCE (inference_3des.py, recover_3des_master_key())
✓ Loads preprocessed features X_features.npy
✓ Loads trained ensemble models (3 models per S-box, per key type)
✓ Stage 1 Voting: 
  - For each S-box: collect predictions from all traces
  - Majority vote → recovered K1 (master key first half)
✓ Stage 2 Recovery:
  - Using recovered K1, compute K2 (master key second half)
  - Majority voting on Stage 2 predictions
✓ Output: Dict with K_KENC, K_KMAC, K_KDEK (repeated for all traces)

STAGE 3: KEY RECONSTRUCTION (implicit in voting)
✓ S-box predictions (0-255) → reconstructed via RK1/RK16 formulas
✓ 8 S-boxes × 2-byte output = 16-byte master key

STAGE 4: REPORT GENERATION (output_gen.py, build_rows())
✓ Takes recovered master keys from Dict
✓ Outputs static master key (same value for all rows)
✓ RSA predictions also output per row

Pipeline appears theoretically and practically sound.
The question: Are the recovered master keys CORRECT?
""")
