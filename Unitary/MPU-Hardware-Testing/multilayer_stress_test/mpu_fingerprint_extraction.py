# -*- coding: utf-8 -*-
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
    # Reverse bitstring so index i = physical qubit i
    return np.array([int(b) for b in reversed(bitstring)], dtype=np.int32)

def shannon_entropy(p):
    return -sum(v * np.log2(v) for v in p.values() if v > 0)

def mutual_information(p, i, j):
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

def correlation(p, i, j):
    xs, ys, ws = [], [], []
    for bitstring, prob in p.items():
        b = bitstring_to_array(bitstring)
        xs.append(b[i])
        ys.append(b[j])
        ws.append(prob)
    xs = np.array(xs)
    ys = np.array(ys)
    ws = np.array(ws)
    Ex = np.sum(xs * ws)
    Ey = np.sum(ys * ws)
    Exy = np.sum(xs * ys * ws)
    return Exy - Ex * Ey

def mi_matrix(p, N):
    MI = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                MI[i, j] = mutual_information(p, i, j)
    return MI

def corr_matrix(p, N):
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                C[i, j] = correlation(p, i, j)
    return C

# ============================================================
# Bond-profile extraction (MPU-native fingerprint)
# ============================================================

def bond_profile(MI):
    """
    MI-cut profile:
    cut k = sum of MI[i,j] for i < k and j >= k
    """
    N = MI.shape[0]
    profile = []
    for k in range(1, N):
        val = np.sum(MI[:k, k:])
        profile.append(float(val))
    return profile

# ============================================================
# Collapse-geometry signature (same method as earlier)
# ============================================================

def collapse_geometry(MI, C):
    """
    Flatten MI and Corr, concatenate, normalize.
    This matches your earlier collapse-geometry signature logic.
    """
    v = np.concatenate([MI.flatten(), C.flatten()])
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()

# ============================================================
# Load JSON helper
# ============================================================

def load_counts(filename):
    with open(filename, "r") as f:
        return json.load(f)["counts"]

# ============================================================
# Main MPU fingerprint extractor
# ============================================================

def extract_mpu_fingerprints(N=6):

    mpu_files = {
        "000000": "ibm_run_multilayer_N6_000000.json",
        "001111": "ibm_run_multilayer_N6_001111.json",
        "010101": "ibm_run_multilayer_N6_010101.json",
        "0p0p0p": "ibm_run_multilayer_N6_0p0p0p.json",
        "111000": "ibm_run_multilayer_N6_111000.json",
        "allplus": "ibm_run_multilayer_N6_allplus.json",
        "plusplus": "ibm_run_multilayer_N6_plusplus.json"
    }

    fingerprints = {}

    for label, fname in mpu_files.items():
        print(f"\n=== Extracting MPU fingerprint for {label} ===")

        counts = load_counts(fname)
        p = normalize_counts(counts)

        H = shannon_entropy(p)
        MI = mi_matrix(p, N)
        C = corr_matrix(p, N)
        BP = bond_profile(MI)
        CG = collapse_geometry(MI, C)

        fingerprints[label] = {
            "entropy": float(H),
            "MI_matrix": MI.tolist(),
            "corr_matrix": C.tolist(),
            "bond_profile": BP,
            "collapse_geometry": CG
        }

        print(f"Entropy: {H:.4f}")
        print(f"Bond profile: {BP}")
        print(f"Collapse geometry length: {len(CG)}")

    # Save all fingerprints
    with open("mpu_multilayer_fingerprints.json", "w") as f:
        json.dump(fingerprints, f, indent=2)

    print("\nSaved MPU fingerprints → mpu_multilayer_fingerprints.json")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    extract_mpu_fingerprints(N=6)
