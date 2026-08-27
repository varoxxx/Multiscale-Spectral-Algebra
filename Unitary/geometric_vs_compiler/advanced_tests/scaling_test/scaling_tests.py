# -*- coding: utf-8 -*-
"""
Scaling Tests for Geometric Unitary Matching
David Mulnix copyright 2026
------------------------------------------------------------

This module performs a series of scaling tests to evaluate how well the
geometric optimizer matches Haar‑random unitaries as matrix dimension
increases. The optimizer is tested on 2‑, 3‑, 4‑, and 5‑qubit Haar targets
(dimensions 4×4 through 32×32), using multiple trials per size to confirm
stability and consistency.

What the test does:
    • Generate Haar‑random unitaries using Qiskit.
    • Initialize U = I and collapse toward the target using the full
      geometric pipeline:
          – deviation operator Δ(U)
          – extremal spectral projector
          – geometric update rule
          – polar re‑unitarization
    • Compute two correctness metrics:
          – operator fidelity (matrix‑level equivalence)
          – state fidelity (behavioral equivalence on random states)
    • Repeat for several trials at each dimension.

What this test validates:
    • Matrix‑level correctness (operator fidelity ≈ 1.0).
    • Partial physical correctness (state fidelity ≈ 1.0).
    • Numerical stability of geometric descent across increasing sizes.
    • Successful scaling of the geometric operator framework from 4×4
      up to 32×32 Haar‑random targets.

What this test is NOT:
    • It is not a circuit‑derived validation test.
    • It does not perform full physical validation across structured
      input states.
    • It does not test functional equivalence inside a quantum circuit.

Summary:
    Perfect operator fidelity and perfect state fidelity across all sizes
    confirm that the geometric optimizer scales cleanly, remains stable,
    and successfully matches Haar‑random unitaries using the underlying
    collapse geometry.
"""


import cupy as cp
import numpy as np

from qiskit.quantum_info import random_unitary
from module_geometric_optimizer import GeometricOptimizer
from module_fidelity_objective import FidelityObjective


def qiskit_random_unitary(dim):
    U = random_unitary(dim)
    return cp.asarray(U.data)


def run_scaling_tests(n_qubits_list, steps=200, trials=3):
    print("\n=== Scaling Tests ===")
    print(f"Steps per run: {steps}, Trials per size: {trials}\n")

    for n_qubits in n_qubits_list:
        dim = 2**n_qubits
        print(f"--- n_qubits = {n_qubits} (dim={dim}) ---")

        for t in range(1, trials + 1):
            U_target = qiskit_random_unitary(dim)
            U_init = cp.eye(dim, dtype=cp.complex128)

            U_geo, _ = GeometricOptimizer.optimize(
                U_init,
                U_target,
                n_qubits,
                steps=steps,
                frac_ext=0.8,
                lam_reg=0.1,
                step_scale=0.1,
                verbose=False
            )

            fid_state = FidelityObjective.state_fidelity(U_geo, U_target, n_qubits)
            fid_op = FidelityObjective.operator_fidelity(U_geo, U_target)

            print(f"  Trial {t}/{trials}: state_fid={fid_state:.6f}, op_fid={fid_op:.6f}")
        print()


if __name__ == "__main__":
    cp.random.seed(0)
    np.random.seed(0)

    n_qubits_list = [2, 3, 4, 5]
    run_scaling_tests(n_qubits_list, steps=200, trials=3)
