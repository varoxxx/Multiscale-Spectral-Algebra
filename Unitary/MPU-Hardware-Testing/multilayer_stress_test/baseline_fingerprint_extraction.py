# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

import json
import numpy as np

# ============================================================
# Core utilities (same as MPU fingerprint script)
# ============================================================

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def bitstring_to_array(bitstring):
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
# Bond-profile extraction (same as MPU fingerprint script)
# ============================================================

def bond_profile(MI):
    N = MI.shape[0]
    profile = []
    for k in range(1, N):
        val = np.sum(MI[:k, k:])
        profile.append(float(val))
    return profile

# ============================================================
# Collapse-geometry signature (same as MPU fingerprint script)
# ============================================================

def collapse_geometry(MI, C):
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
# Main baseline fingerprint extractor
# ============================================================

def extract_baseline_fingerprints(N=6):

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

    fingerprints = {}

    for gate in gates:
        fingerprints[gate] = {}

        for state in input_states:
            fname = f"{gate}_{state}.json"
            print(f"\n=== Extracting baseline fingerprint: {gate.upper()} |{state}⟩ ===")

            counts = load_counts(fname)
            p = normalize_counts(counts)

            H = shannon_entropy(p)
            MI = mi_matrix(p, N)
            C = corr_matrix(p, N)
            BP = bond_profile(MI)
            CG = collapse_geometry(MI, C)

            fingerprints[gate][state] = {
                "entropy": float(H),
                "MI_matrix": MI.tolist(),
                "corr_matrix": C.tolist(),
                "bond_profile": BP,
                "collapse_geometry": CG
            }

            print(f"Entropy: {H:.4f}")
            print(f"Bond profile: {BP}")
            print(f"Collapse geometry length: {len(CG)}")

    # Save all baseline fingerprints
    with open("baseline_fingerprints.json", "w") as f:
        json.dump(fingerprints, f, indent=2)

    print("\nSaved baseline fingerprints → baseline_fingerprints.json")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    extract_baseline_fingerprints(N=6)
