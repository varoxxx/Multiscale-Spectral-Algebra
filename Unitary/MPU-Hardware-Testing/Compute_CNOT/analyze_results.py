
"""
David Mulnix copyright 2026
"""

import json
import glob

# ============================================================
# 2‑bit CNOT reversible function
# ============================================================

def int_to_bits(x, N=2):
    return format(x, f"0{N}b")

def bits_to_int(b):
    return int(b, 2)

def reversible_function_cnot(x):
    """
    CNOT truth table (control = qubit 1, target = qubit 0)
    Qiskit little-endian convention:
        |q1 q0>

    Mapping:
        00 -> 00
        01 -> 01
        10 -> 11
        11 -> 10
    """
    mapping = {
        0: 0,  # 00 -> 00
        1: 1,  # 01 -> 01
        2: 3,  # 10 -> 11
        3: 2   # 11 -> 10
    }
    return mapping[x]

# ============================================================
# Verification logic
# ============================================================

def verify_json_file(fname):
    print(f"\n=== Verifying {fname} ===")

    with open(fname, "r") as f:
        data = json.load(f)

    label = data["input_state"]   # e.g., "00"
    counts = data["counts"]       # hardware counts

    # Convert label (bitstring) to integer
    x = bits_to_int(label)

    # Compute expected reversible output
    y_expected = reversible_function_cnot(x)
    b_expected = int_to_bits(y_expected)

    # Find most probable output from hardware
    b_observed = max(counts.items(), key=lambda kv: kv[1])[0]

    print(f"Input state:        |{label}>")
    print(f"Expected output:    |{b_expected}>")
    print(f"Observed top state: |{b_observed}>")

    if b_observed == b_expected:
        print("RESULT: PASS ✓")
        return True
    else:
        print("RESULT: FAIL ✗")
        return False

# ============================================================
# Main: verify all JSON files matching your naming pattern
# ============================================================

if __name__ == "__main__":
    files = glob.glob("ibm_run_multilayer_N2_*.json")

    if not files:
        print("No JSON files found.")
        exit(1)

    total = len(files)
    passed = 0

    for fname in files:
        if verify_json_file(fname):
            passed += 1

    print("\n====================================")
    print(f"Verification complete: {passed}/{total} tests passed")
    print("====================================")
