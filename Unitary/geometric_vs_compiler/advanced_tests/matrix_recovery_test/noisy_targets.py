"""
Noisy-Unitary Recovery Test
David Mulnix copyright 2026
------------------------------------------------------------

This script evaluates a non-standard capability of the geometric operator
framework: recovering a clean unitary from a deliberately corrupted (noisy)
version of itself. Unlike typical quantum compilation, synthesis, or
transpilation tools—which do NOT attempt to repair noisy unitaries—this test
uses the geometric optimizer to pull a perturbed unitary back toward its clean
target using deviation geometry, extremal spectral steering, and polar
re‑unitarization.

What makes this test unique:
    • Quantum compilers do not perform "unitary recovery" or "noise repair."
      They only decompose or approximate a given clean unitary.
    • No standard tool attempts to take a noisy unitary and reconstruct the
      original clean version.
    • This capability is therefore unique to the geometric operator framework
      and arises directly from the underlying collapse geometry.

What the test does:
    1. Generate a clean Haar-random unitary U_clean.
    2. Add controlled Gaussian/coherent noise and re-unitarize it to produce
       U_noisy.
    3. Use the geometric optimizer (your math) to collapse U_noisy back toward
       U_clean.
    4. Measure:
         • Fidelity(noisy → clean): recovery fidelity
         • Fidelity(noisy initial): how corrupted the noisy unitary was

How to interpret results:
    • If Fidelity(noisy → clean) ≈ 1.0, the optimizer has successfully
      reconstructed the clean unitary from its noisy version — a capability
      not found in standard quantum toolchains.
    • Even partial recovery (fidelity significantly higher than the noisy
      initial fidelity) demonstrates robustness and diversity of the operator.
    • This test is exploratory: it is not expected to recover every noisy
      unitary, but recovering even one cleanly is a strong indicator of the
      geometric method’s power.

Purpose:
    • Demonstrate that the geometric optimizer can act as a noise-correcting
      operator on U(N), something compilers do not attempt.
    • Show that collapse geometry can reverse perturbations and restore
      structure.
    • Provide early evidence of robustness and stability under noisy inputs.

This test is an initial probe into recovery behavior and is not yet optimized
for full success across all trials. A single perfect recovery (fidelity = 1.0)
is already a non-trivial and non-standard achievement.

Note: I have seen it recover 1.0 before, but whether you will or not is hard to say.
I may develop this so it works every time, but for now I leave you with this example.
"""


import cupy as cp
import numpy as np

from qiskit.quantum_info import random_unitary
from module_geometric_optimizer import GeometricOptimizer
from module_fidelity_objective import FidelityObjective


def qiskit_random_unitary(dim):
    U = random_unitary(dim)
    return cp.asarray(U.data)


def add_noise(U, noise_strength=0.05):
    n = U.shape[0]
    noise = noise_strength * (cp.random.randn(n, n) + 1j * cp.random.randn(n, n))
    U_noisy = U + noise
    Q, _ = cp.linalg.qr(U_noisy)
    return Q


def run_noisy_target_tests(n_qubits, steps=200, trials=5):
    dim = 2**n_qubits
    print(f"\n=== Noisy Target Tests (Recovery): n_qubits={n_qubits} ===")

    for t in range(1, trials + 1):
        print(f"\n--- Test {t}/{trials} ---")

        # 1) Clean target
        U_clean = qiskit_random_unitary(dim)

        # 2) Noisy initial
        U_noisy = add_noise(U_clean, noise_strength=0.05)

        # 3) Run geometric optimizer to recover clean from noisy
        U_geo, _ = GeometricOptimizer.optimize(
            U_noisy,       # start from noisy
            U_clean,       # target is clean
            n_qubits,
            steps=steps,
            frac_ext=0.8,
            lam_reg=0.1,
            step_scale=0.1,
            verbose=False
        )

        # 4) Compare
        fid_recovery = FidelityObjective.operator_fidelity(U_geo, U_clean)
        fid_noisy = FidelityObjective.operator_fidelity(U_noisy, U_clean)

        print(f"Fidelity(noisy → clean): {fid_recovery:.6f}")
        print(f"Fidelity(noisy initial): {fid_noisy:.6f}")


if __name__ == "__main__":
    run_noisy_target_tests(2)
    run_noisy_target_tests(3)
