"""
David Mulnix copyright 2026
"""

import json
import sys
import numpy as np
from math import log2
from pprint import pprint

###############################################################################
# Utility Functions
###############################################################################

def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON: {e}")
        sys.exit(1)

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}, total

def shannon_entropy(dist):
    return -sum(p * log2(p) for p in dist.values() if p > 0)

def kl_divergence(p, q):
    kl = 0.0
    for k in p:
        if p[k] > 0:
            kl += p[k] * log2(p[k] / q[k])
    return kl

def hellinger_distance(p, q):
    return np.sqrt(0.5 * sum((np.sqrt(p[k]) - np.sqrt(q[k]))**2 for k in p))

def bitstring_to_array(bitstring):
    return np.array([int(b) for b in bitstring])

###############################################################################
# Mutual Information
###############################################################################

def mutual_information(dist, n_qubits):
    bitstrings = list(dist.keys())
    probs = np.array([dist[b] for b in bitstrings])
    X = np.array([bitstring_to_array(b) for b in bitstrings])

    MI = np.zeros((n_qubits, n_qubits))

    for i in range(n_qubits):
        for j in range(n_qubits):
            if i == j:
                MI[i, j] = 0
                continue

            joint = {}
            for idx, b in enumerate(bitstrings):
                key = (X[idx, i], X[idx, j])
                joint[key] = joint.get(key, 0) + probs[idx]

            pi = {0: 0, 1: 0}
            pj = {0: 0, 1: 0}
            for (bi, bj), p in joint.items():
                pi[bi] += p
                pj[bj] += p

            mi = 0
            for (bi, bj), p in joint.items():
                if p > 0:
                    mi += p * log2(p / (pi[bi] * pj[bj]))
            MI[i, j] = mi

    return MI

###############################################################################
# Main Analysis
###############################################################################

def main():
    if len(sys.argv) < 2:
        print("Usage: python mpu_analysis.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    print(f"\n=== MPU HARDWARE ANALYSIS ===")
    print(f"Loading: {json_file}")

    data = safe_load_json(json_file)

    ###########################################################################
    # Extract counts
    ###########################################################################
    counts = data["counts"]
    dist, total_shots = normalize_counts(counts)

    print("\n=== Counts Summary ===")
    print(f"Total shots: {total_shots}")
    print(f"Unique bitstrings: {len(dist)}")

    ###########################################################################
    # Entropy
    ###########################################################################
    entropy = shannon_entropy(dist)
    print(f"\nShannon Entropy: {entropy:.6f} bits")

    ###########################################################################
    # KL Divergence vs Uniform
    ###########################################################################
    n_qubits = len(next(iter(counts.keys())))
    uniform = {b: 1 / (2**n_qubits) for b in dist}
    kl = kl_divergence(dist, uniform)
    print(f"KL Divergence vs Uniform: {kl:.6f}")

    ###########################################################################
    # Hellinger Distance vs Uniform
    ###########################################################################
    hell = hellinger_distance(dist, uniform)
    print(f"Hellinger Distance vs Uniform: {hell:.6f}")

    ###########################################################################
    # Mutual Information Matrix
    ###########################################################################
    print("\n=== Mutual Information Matrix ===")
    MI = mutual_information(dist, n_qubits)
    total_mi = np.sum(MI)

    print(f"Total MI: {total_mi:.6f}\n")
    print("MI Matrix:")
    for row in MI:
        print(" ".join(f"{v:.4f}" for v in row))

    ###########################################################################
    # Noise Diagnostics
    ###########################################################################
    print("\n=== Noise Diagnostics ===")
    print(f"Depth: {data['depth']}")
    print(f"Gate Counts:")
    pprint(data["gate_counts"])
    print(f"Coupling Map Size: {len(data['coupling_map'])}")
    print(f"Basis Gates: {data['basis_gates']}")
    print(f"T1 entries: {len(data['t1'])}")
    print(f"T2 entries: {len(data['t2'])}")
    print(f"Readout error entries: {len(data['readout'])}")
    print(f"CX gate error: {data['cx_gate_error']}")

    ###########################################################################
    # Gate Error Table Summary
    ###########################################################################
    print("\n=== Gate Error Table Summary ===")
    gate_errors = data["gate_errors"]
    print(f"Total gate error entries: {len(gate_errors)}")
    first_key = next(iter(gate_errors.keys()))
    print(f"Example gate error entry ({first_key}):")
    pprint(gate_errors[first_key])

    ###########################################################################
    # Save analysis results
    ###########################################################################
    analysis_output = {
        "entropy": entropy,
        "kl_divergence": kl,
        "hellinger_distance": hell,
        "total_mutual_information": float(total_mi),
        "mutual_information_matrix": MI.tolist(),
        "n_qubits": n_qubits,
        "total_shots": total_shots,
        "unique_bitstrings": len(dist)
    }

    out_file = json_file.replace(".json", "_analysis.json")
    with open(out_file, "w") as f:
        json.dump(analysis_output, f, indent=2)

    print(f"\nSaved analysis results to: {out_file}")
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
