# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

import json
import numpy as np

# ============================================================
# Load fingerprint JSONs
# ============================================================

def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)

mpu_fp = load_json("mpu_multilayer_fingerprints.json")
base_fp = load_json("baseline_fingerprints.json")

# ============================================================
# Distance metrics
# ============================================================

def l2(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))

def matrix_l2(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    return float(np.linalg.norm(A - B))

# ============================================================
# Comparison routine
# ============================================================

def compare_mpu_vs_baseline(N=6):

    input_states = [
        "000000",
        "001111",
        "010101",
        "0p0p0p",
        "111000",
        "allplus",
        "plusplus"
    ]

    gates = ["cz", "cnot", "qft_like"]

    print("\n==============================")
    print("MPU vs BASELINE COMPARISON")
    print("==============================\n")

    for state in input_states:
        print(f"\n=== State {state} ===")

        mpu = mpu_fp[state]

        for gate in gates:
            base = base_fp[gate][state]

            bp_dist = l2(mpu["bond_profile"], base["bond_profile"])
            cg_dist = l2(mpu["collapse_geometry"], base["collapse_geometry"])
            mi_dist = matrix_l2(mpu["MI_matrix"], base["MI_matrix"])
            corr_dist = matrix_l2(mpu["corr_matrix"], base["corr_matrix"])

            print(f"\n--- {gate.upper()} comparison ---")
            print(f"Bond-profile L2 distance:       {bp_dist:.6f}")
            print(f"Collapse-geometry L2 distance:  {cg_dist:.6f}")
            print(f"MI-matrix L2 distance:          {mi_dist:.6f}")
            print(f"Corr-matrix L2 distance:        {corr_dist:.6f}")

        print("\n----------------------------------")

    print("\nComparison complete.")
    print("==================================")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    compare_mpu_vs_baseline(N=6)
