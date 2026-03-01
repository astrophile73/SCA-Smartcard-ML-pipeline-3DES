"""
Launcher wrapper to run the local 3DES pipeline from this repo.

Usage examples (from D:\\Projects\\SCA-Smartcard-ML-pipeline-3DES):
  python main.py --mode train
  python main.py --mode train --epochs 50
  python main.py --mode train --scan_type 3des --card_type universal
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.join(BASE_DIR, "pipeline-code")
REAL_MAIN = os.path.join(CODE_ROOT, "main.py")
NEW_ROOT = os.path.join(BASE_DIR, "3des-pipeline")
DEFAULT_XLSX = os.path.join(BASE_DIR, "KALKi TEST CARD.xlsx")


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _ensure_arg(argv: list[str], flag: str, value: str) -> None:
    if not _has_flag(argv, flag):
        argv.extend([flag, value])


def _default_input_dir() -> str:
    candidates = [
        os.path.join(NEW_ROOT, "Input"),
        os.path.join(NEW_ROOT, "Input1"),
        os.path.join(BASE_DIR, "Input"),
        os.path.join(BASE_DIR, "Input1"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _build_command(user_args: list[str]) -> list[str]:
    args = list(user_args)

    # If no mode is provided, default to train.
    if "--mode" not in args:
        args.extend(["--mode", "train"])

    # Always keep runtime artifacts in the 3DES repo unless overridden.
    _ensure_arg(args, "--processed_dir", os.path.join(NEW_ROOT, "Processed", "run_manual"))
    _ensure_arg(args, "--opt_dir", os.path.join(NEW_ROOT, "Optimization", "run_manual"))
    _ensure_arg(args, "--output_dir", os.path.join(NEW_ROOT, "Output", "run_manual"))
    _ensure_arg(args, "--model_root", os.path.join(NEW_ROOT, "models", "run_manual"))

    # Keep input local to this repo unless overridden.
    _ensure_arg(args, "--input_dir", _default_input_dir())

    # 3DES defaults for your workflow unless overridden.
    _ensure_arg(args, "--scan_type", "3des")
    _ensure_arg(args, "--card_type", "universal")
    _ensure_arg(args, "--file_pattern", "traces_data_*.npz")

    # Enable external labels for train runs unless user provided alternatives.
    if "train" in args or ("--mode" in args and args[args.index("--mode") + 1] == "train"):
        if not _has_flag(args, "--enable_external_labels"):
            args.append("--enable_external_labels")
        _ensure_arg(args, "--label_map_xlsx", DEFAULT_XLSX)
        if not _has_flag(args, "--strict_label_mode"):
            args.append("--strict_label_mode")

    return [sys.executable, REAL_MAIN, *args]


def main() -> int:
    if not os.path.exists(REAL_MAIN):
        print(f"ERROR: local pipeline not found: {REAL_MAIN}")
        return 2

    os.makedirs(os.path.join(NEW_ROOT, "Processed"), exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "Optimization"), exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "Output"), exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "models"), exist_ok=True)

    cmd = _build_command(sys.argv[1:])
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    return subprocess.call(cmd, cwd=CODE_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
