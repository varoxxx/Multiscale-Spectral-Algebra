# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

import json
import numpy as np
import os

# ============================================================
# Core math utilities (same as MPU analysis)
# ============================================================

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def bitstring_to_array(bitstring):
    # Reverse bitstring so index i corresponds to physical qubit i
    return np.array([int(b) for b in reversed(bitstring)], dtype=np.int32)

# ============================================================
# Baseline format checks (must match MPU format)
# ============================================================

def check_baseline_format(filename, N=6):
    with open(filename, "r") as f:
        data = json.load(f)

    print(f"\n=== Checking baseline file: {filename} ===")

    # 1. Must contain ONLY "counts"
    keys = set(data.keys())
    if keys != {"counts"}:
        print(f"FAIL: baseline file contains extra fields: {keys}")
        return False
    print("OK: baseline contains only 'counts'")

    counts = data["counts"]

    # 2. Bitstrings must be length N and only contain 0/1
    for b in counts.keys():
        if len(b) != N:
            print(f"FAIL: bitstring length mismatch: {b} (len={len(b)})")
            return False
        if any(ch not in "01" for ch in b):
            print(f"FAIL: invalid character in bitstring: {b}")
            return False
    print("OK: bitstrings are valid 6-bit binary strings")

    # 3. Normalization check
    p = normalize_counts(counts)
    s = sum(p.values())
    if abs(s - 1.0) > 1e-6:
        print(f"FAIL: probabilities do not normalize to 1 (sum={s})")
        return False
    print("OK: probabilities normalize correctly")

    print("PASS: baseline file matches MPU format")
    return True

# ============================================================
# MAIN: check all baseline files
# ============================================================

if __name__ == "__main__":
    baseline_files = [
        "cz_plusplus.json",
        "cz_plus0.json",
        "cz_0plus.json",
        "cz_allplus.json",
        "cnot_plusplus.json",
        "cnot_plus0.json",
        "cnot_0plus.json",
        "cnot_allplus.json",
        "qft_like_plusplus.json",
        "qft_like_plus0.json",
        "qft_like_0plus.json",
        "qft_like_allplus.json",
    ]

    overall_ok = True
    for fname in baseline_files:
        if os.path.exists(fname):
            ok = check_baseline_format(fname, N=6)
            overall_ok = overall_ok and ok
        else:
            print(f"WARNING: baseline file missing: {fname}")

    print("\n====================================================")
    print(f"Overall baseline format: {'PASS' if overall_ok else 'FAIL'}")
    print("====================================================")
