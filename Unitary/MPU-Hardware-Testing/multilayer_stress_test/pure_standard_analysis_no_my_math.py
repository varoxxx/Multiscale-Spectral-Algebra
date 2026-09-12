# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

import json
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ============================================================
# Load raw counts
# ============================================================

def load_counts(fname):
    with open(fname, "r") as f:
        return json.load(f)["counts"]

def normalize(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

# ============================================================
# Standard statistical distances
# ============================================================

def kl(p, q):
    eps = 1e-12
    return sum(p[k] * np.log2((p[k] + eps) / (q.get(k, eps))) for k in p)

def hellinger(p, q):
    return np.sqrt(sum((np.sqrt(p[k]) - np.sqrt(q.get(k, 0)))**2 for k in p)) / np.sqrt(2)

def tv(p, q):
    return 0.5 * sum(abs(p[k] - q.get(k, 0)) for k in p)

def chi_square(p, q):
    eps = 1e-12
    return sum((p[k] - q.get(k, eps))**2 / q.get(k, eps) for k in p)

def cross_entropy(p, q):
    eps = 1e-12
    return -sum(p[k] * np.log2(q.get(k, eps)) for k in p)

def overlap(p, q):
    return sum(min(p[k], q.get(k, 0)) for k in p)

def support_size(p):
    return sum(1 for k in p if p[k] > 0)

# ============================================================
# Random circuit generator
# ============================================================

def random_counts(N=6, shots=4096, depth=6):
    qc = QuantumCircuit(N, N)
    for _ in range(depth):
        for q in range(N):
            qc.rx(np.random.uniform(0, 2*np.pi), q)
        for q in range(N - 1):
            qc.cx(q, q + 1)
    qc.measure(range(N), range(N))
    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    return normalize(result.get_counts())

# ============================================================
# Main standard analysis
# ============================================================

def standard_analysis():

    input_states = [
        "000000",
        "001111",
        "010101",
        "0p0p0p",
        "111000",
        "allplus",
        "plusplus"
    ]

    baselines = ["cz", "cnot", "qft_like"]

    print("\n====================================")
    print("PURE STANDARD ANALYSIS (NO GEOMETRY)")
    print("====================================\n")

    for state in input_states:
        print(f"\n=== STATE {state} ===")

        # Load MPU raw distribution
        mpu = normalize(load_counts(f"ibm_run_multilayer_N6_{state}.json"))

        # Random circuit distribution
        rnd = random_counts()

        for gate in baselines:
            base = normalize(load_counts(f"{gate}_{state}.json"))

            print(f"\n--- {gate.upper()} comparison ---")

            print(f"KL divergence:        {kl(mpu, base):.6f}")
            print(f"Hellinger distance:   {hellinger(mpu, base):.6f}")
            print(f"Total variation:      {tv(mpu, base):.6f}")
            print(f"Chi-square:           {chi_square(mpu, base):.6f}")
            print(f"Cross entropy:        {cross_entropy(mpu, base):.6f}")
            print(f"Overlap:              {overlap(mpu, base):.6f}")
            print(f"Support size (MPU):   {support_size(mpu)}")
            print(f"Support size (BASE):  {support_size(base)}")

        print("\n--- RANDOM CIRCUIT comparison ---")
        print(f"KL divergence:        {kl(mpu, rnd):.6f}")
        print(f"Hellinger distance:   {hellinger(mpu, rnd):.6f}")
        print(f"Total variation:      {tv(mpu, rnd):.6f}")
        print(f"Chi-square:           {chi_square(mpu, rnd):.6f}")
        print(f"Cross entropy:        {cross_entropy(mpu, rnd):.6f}")
        print(f"Overlap:              {overlap(mpu, rnd):.6f}")
        print(f"Support size (MPU):   {support_size(mpu)}")
        print(f"Support size (RAND):  {support_size(rnd)}")

        print("\n----------------------------------")

    print("\nStandard analysis complete.")
    print("====================================")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    standard_analysis()
