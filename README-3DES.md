# Smartcard 3DES Side-Channel Pipeline (CPA + ML)

## Description
This project is a reproducible, headless side-channel analysis (SCA) workflow for smartcard traces captured with ChipWhisperer Husky. It focuses on building a clean 3DES pipeline that ingests raw traces + metadata, performs alignment and point-of-interest (POI) selection, establishes a CPA baseline, and then uses a PyTorch model to improve key-recovery robustness. The pipeline is designed for Linux/GPU execution and produces a single spreadsheet report per run.

## Objective
The objective is to recover 2-key 3DES (TDEA) session key material (e.g., KENC/KMAC/KDEK depending on the operation and dataset) directly from measured traces, using only the information present in each dataset. The pipeline must support datasets that provide different metadata fields (e.g., labeled Mastercard-style traces with T_DES_* keys, and GreenVisa-style traces providing Track2, UN, and ATC).

Authorized-use only: this repo is intended for controlled lab data and approved testing.

## Client Requirements (Acceptance Criteria)
- End-to-end, fully scripted workflow for headless Linux workstation execution (modern GPU available).
- Preprocess and align raw Husky traces (power and any available metadata).
- Baseline CPA to identify leakage/POIs and to benchmark ML improvements.
- PyTorch-based ML stage (CNN/RNN/hybrid as justified) that improves recovery success over CPA baseline.
- Output recovered keys with confidence metrics.
- Clear documentation so the workflow can be re-run and extended later.
- Input directory may contain 3DES traces, RSA traces, or both; the final integrated workflow should run what is available and leave blanks for unavailable outputs.
  - Current repo scope: 3DES pipeline is implemented first; RSA/PIN extraction is planned as a later integration.

## Repository Layout
This repo keeps large artifacts out of git. Only empty folders are committed.

Input/          # put raw trace datasets here (NPZ/CSV exports)
Output/         # final reports
models/         # trained model checkpoints (not committed)
Optimization/   # POIs, reference traces, normalization stats (not committed)
Processed/      # extracted features + metadata for training/attack (not committed)

## Supported Metadata (3DES)
The pipeline supports multiple dataset "shapes" without removing any existing logic:
- Labeled 3DES datasets (training/validation possible):
  - trace_data, Track2, ATC, and T_DES_KENC/KMAC/KDEK (or equivalent)
- Unlabeled 3DES datasets (inference-only unless validation is provided):
  - trace_data, Track2, ATC, UN
  - The pipeline derives the per-trace 8-byte input block from available fields (ACR when present, otherwise UN+ATC).

## Future Work (RSA/PIN)
RSA CRT recovery and PIN-block extraction will be integrated later as a second sub-pipeline. This requires datasets that include appropriate per-trace metadata (e.g., APDU/ACR streams and/or dataset-internal labels for supervised training).
