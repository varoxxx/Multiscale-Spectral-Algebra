"""
David Mulnix copyright 2026
"""

import json
import glob

# ============================================================
# 2‑bit reversible function (same mapping used in your MPU test)
# ============================================================

def int_to_bits(x, N=2):
    return format(x, f"0{N}b")

def bits_to_int(b):
    return int(b, 2)

def reversible_function_2bit(x):
    """
    2‑bit reversible mapping:
        00 -> 01
        01 -> 11
        10 -> 00
        11 -> 10
    """
    mapping = {
        0: 1,  # 00 -> 01
        1: 3,  # 01 -> 11
        2: 0,  # 10 -> 00
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
    y_expected = reversible_function_2bit(x)
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
