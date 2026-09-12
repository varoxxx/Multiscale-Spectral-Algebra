# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

import json
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ============================================================
# Load fingerprints
# ============================================================

def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)

mpu_fp = load_json("mpu_multilayer_fingerprints.json")
base_fp = load_json("baseline_fingerprints.json")

# ============================================================
# Utility: normalize dict counts
# ============================================================

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

# ============================================================
# Statistical distances
# ============================================================

def kl_divergence(p, q):
    eps = 1e-12
    kl = 0.0
    for k in p:
        kl += p[k] * np.log2((p[k] + eps) / (q.get(k, eps)))
    return float(kl)

def hellinger(p, q):
    s = 0.0
    for k in p:
        s += (np.sqrt(p[k]) - np.sqrt(q.get(k, 0)))**2
    return float(np.sqrt(s) / np.sqrt(2))

def total_variation(p, q):
    tv = 0.5 * sum(abs(p[k] - q.get(k, 0)) for k in p)
    return float(tv)

def chi_square(p, q):
    eps = 1e-12
    chi = 0.0
    for k in p:
        chi += (p[k] - q.get(k, eps))**2 / (q.get(k, eps))
    return float(chi)

# ============================================================
# Random circuit generator
# ============================================================

def random_circuit_counts(N=6, shots=2048, depth=6):
    qc = QuantumCircuit(N, N)
    for _ in range(depth):
        for q in range(N):
            qc.rx(np.random.uniform(0, 2*np.pi), q)
        for q in range(N - 1):
            qc.cx(q, q + 1)
    qc.measure(range(N), range(N))
    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    return normalize_counts(result.get_counts())

# ============================================================
# Main comparison
# ============================================================

def standard_analysis(N=6):

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

    print("\n====================================")
    print("STANDARD ANALYSIS: MPU vs BASELINES")
    print("====================================\n")

    for state in input_states:
        print(f"\n=== State {state} ===")

        # MPU distribution
        mpu_MI = np.array(mpu_fp[state]["MI_matrix"])
        mpu_entropy = mpu_fp[state]["entropy"]

        # Reconstruct MPU distribution from MI fingerprint file
        # (We stored only MI/corr/entropy; reload raw counts)
        with open(f"ibm_run_multilayer_N6_{state}.json", "r") as f:
            mpu_counts = normalize_counts(json.load(f)["counts"])

        # Random circuit distribution
        rand_counts = random_circuit_counts(N=N)

        for gate in gates:
            with open(f"{gate}_{state}.json", "r") as f:
                base_counts = normalize_counts(json.load(f)["counts"])

            kl = kl_divergence(mpu_counts, base_counts)
            hel = hellinger(mpu_counts, base_counts)
            tv = total_variation(mpu_counts, base_counts)
            chi = chi_square(mpu_counts, base_counts)

            print(f"\n--- {gate.upper()} comparison ---")
            print(f"KL divergence:        {kl:.6f}")
            print(f"Hellinger distance:   {hel:.6f}")
            print(f"Total variation:      {tv:.6f}")
            print(f"Chi-square:           {chi:.6f}")

        # MPU vs Random
        kl_r = kl_divergence(mpu_counts, rand_counts)
        hel_r = hellinger(mpu_counts, rand_counts)
        tv_r = total_variation(mpu_counts, rand_counts)
        chi_r = chi_square(mpu_counts, rand_counts)

        print("\n--- RANDOM CIRCUIT comparison ---")
        print(f"KL divergence:        {kl_r:.6f}")
        print(f"Hellinger distance:   {hel_r:.6f}")
        print(f"Total variation:      {tv_r:.6f}")
        print(f"Chi-square:           {chi_r:.6f}")

        print("\n----------------------------------")

    print("\nStandard analysis complete.")
    print("====================================")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    standard_analysis(N=6)
