"""
David Mulnix copyright 2026
"""


import json
import os

# ============================================================
# Core utilities
# ============================================================

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def check_bitstrings(counts, N):
    for b in counts.keys():
        if len(b) != N:
            return False, f"Bitstring length mismatch: {b} (len={len(b)})"
        if any(ch not in "01" for ch in b):
            return False, f"Invalid character in bitstring: {b}"
    return True, "Bitstrings OK"

def check_normalization(p):
    s = sum(p.values())
    return abs(s - 1.0) < 1e-6, f"Normalization sum={s}"

# ============================================================
# MPU format checker (metadata allowed)
# ============================================================

def check_mpu_format(N=6):
    mpu_files = {
        "plusplus":   "ibm_run_multilayer_N6_test_case_1.json",
        "plus0":      "ibm_run_multilayer_N6_test_case_2.json",
        "0plus":      "ibm_run_multilayer_N6_test_case_3.json",
        "allplus":    "ibm_run_multilayer_N6_default.json",
    }

    overall_ok = True

    for state, fname in mpu_files.items():
        print(f"\n=== Checking MPU file for |{state}⟩ ===")

        if not os.path.exists(fname):
            print(f"FAIL: MPU file missing: {fname}")
            overall_ok = False
            continue

        with open(fname, "r") as f:
            data = json.load(f)

        # 1. MPU must contain "counts"
        if "counts" not in data:
            print("FAIL: MPU file missing 'counts' field")
            overall_ok = False
            continue
        else:
            print("OK: MPU file contains 'counts'")

        counts = data["counts"]

        # 2. Bitstrings must be valid
        ok, msg = check_bitstrings(counts, N)
        print(f"{msg}: {'OK' if ok else 'FAIL'}")
        overall_ok = overall_ok and ok

        # 3. Normalization must be correct
        p = normalize_counts(counts)
        ok, msg = check_normalization(p)
        print(f"{msg}: {'OK' if ok else 'FAIL'}")
        overall_ok = overall_ok and ok

        print("PASS: MPU file format is compatible\n")

    print("====================================================")
    print(f"Overall MPU format: {'PASS' if overall_ok else 'FAIL'}")
    print("====================================================")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    check_mpu_format(N=6)
