"""
David Mulnix copyright 2026
"""

import json
import numpy as np

# ============================================================
# Core utilities
# ============================================================

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def bitstring_to_array(bitstring):
    return np.array([int(b) for b in reversed(bitstring)], dtype=np.int32)

def mutual_information(p, i, j, N):
    joint = np.zeros((2, 2))
    for bitstring, prob in p.items():
        b = bitstring_to_array(bitstring)
        joint[b[i], b[j]] += prob
    pi = joint.sum(axis=1)
    pj = joint.sum(axis=0)
    mi = 0.0
    for a in range(2):
        for c in range(2):
            if joint[a, c] > 0 and pi[a] > 0 and pj[c] > 0:
                mi += joint[a, c] * np.log2(joint[a, c] / (pi[a] * pj[c]))
    return mi

def mi_matrix(counts, N):
    p = normalize_counts(counts)
    MI = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                MI[i, j] = mutual_information(p, i, j, N)
    return MI

# ============================================================
# Greedy MI-based qubit mapping
# ============================================================

def compute_mapping(MI_mpu, MI_base):
    N = MI_mpu.shape[0]
    mapping = {}
    used = set()
    for i in range(N):
        best_j = None
        best_score = -1
        for j in range(N):
            if j in used:
                continue
            v_mpu = MI_mpu[i].copy()
            v_base = MI_base[j].copy()
            v_mpu[i] = 0
            v_base[j] = 0
            score = np.dot(v_mpu, v_base)
            if score > best_score:
                best_score = score
                best_j = j
        mapping[i] = best_j
        used.add(best_j)
    return mapping

# ============================================================
# Main alignment check
# ============================================================

def check_alignment():
    N = 6

    # NEW: your multilayer MPU hardware files
    mpu_files = {
        "000000": "ibm_run_multilayer_N6_000000.json",
        "010101": "ibm_run_multilayer_N6_010101.json",
        "001111": "ibm_run_multilayer_N6_001111.json",
        "111000": "ibm_run_multilayer_N6_111000.json",
        "allplus": "ibm_run_multilayer_N6_allplus.json",
        "plusplus": "ibm_run_multilayer_N6_plusplus.json",
        "0p0p0p": "ibm_run_multilayer_N6_0p0p0p.json"
    }

    # Baseline files (same as before)
    baseline_files = {
        "000000": "cz_000000.json",
        "010101": "cz_010101.json",
        "001111": "cz_001111.json",
        "111000": "cz_111000.json",
        "allplus": "cz_allplus.json",
        "plusplus": "cz_plusplus.json",
        "0p0p0p": "cz_0p0p0p.json"
    }

    print("\n=== QUBIT ALIGNMENT CHECK (MI-based) ===")

    for state in mpu_files.keys():
        print(f"\n--- State {state} ---")

        with open(mpu_files[state], "r") as f:
            mpu_counts = json.load(f)["counts"]

        with open(baseline_files[state], "r") as f:
            base_counts = json.load(f)["counts"]

        MI_mpu = mi_matrix(mpu_counts, N)
        MI_base = mi_matrix(base_counts, N)

        mapping = compute_mapping(MI_mpu, MI_base)

        print("MPU → Baseline qubit mapping:")
        for i in range(N):
            print(f"  {i} → {mapping[i]}")

    print("\n=== Alignment check complete ===")

if __name__ == "__main__":
    check_alignment()
