# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026


Large‑Scale Haar Unitary Matching Test
--------------------------------------

This test evaluates the geometric optimizer on a high‑dimensional unitary
matching problem. Qiskit is used only to generate a Haar‑random unitary
matrix of size 2^n × 2^n (e.g., 1024×1024 for n=10). The geometric optimizer
then attempts to collapse the identity matrix toward this target using
spectral steering, extremal‑direction projection, and QR/polar re‑unitarization.

Purpose:
    • Demonstrate that the geometric optimizer scales to large unitaries
      (10 qubits and beyond).
    • Validate numerical stability of the update rule, projector, and
      re‑unitarization at high dimension.
    • Show that the optimizer can match arbitrary Haar‑random unitaries,
      not only structured or circuit‑generated ones.
    • Provide reproducible artifacts: the Qiskit target unitary and the
      geometric result are saved as .npy files for external verification.

What this test is NOT:
    • It is not a circuit‑recovery test.
    • It is not a hardware‑efficient or algebraic‑compiler comparison.
    • It is not demonstrating a Qiskit limitation—Qiskit is only used to
      generate the Haar unitary, which it does correctly.

What the test does:
    1. Generate a Haar‑random unitary U_target using Qiskit.
    2. Save U_target to disk for reproducibility.
    3. Initialize U = I (identity).
    4. Run the geometric optimizer for up to `steps` iterations, with
       early stopping when fidelity reaches 1.0 or updates become negligible.
    5. Save the final geometric unitary U_geo to disk.
    6. Run a second comparison test that loads both matrices and computes:
         • Frobenius difference
         • Operator fidelity
         • Exact equality (np.allclose)

Value of this test:
    • Confirms that the geometric optimizer can operate on large unitaries
      without losing unitarity or numerical stability.
    • Shows that the optimizer can match unstructured Haar targets, implying
      it can also match circuit‑generated unitaries at the same scale.
    • Provides a transparent, verifiable demonstration of geometric descent
      behavior in high‑dimensional U(N).

This test is intended for public release and provides a clear demonstration of
the scalability and stability of the geometric operator framework.


"""

import cupy as cp
import numpy as np
import time


from qiskit.quantum_info import random_unitary
from module_geometric_optimizer import GeometricOptimizer
from module_fidelity_objective import FidelityObjective


def run_large_scale_test(n_qubits=10, steps=600):
    dim = 2**n_qubits
    print(f"\n=== Large-Scale Test: n_qubits={n_qubits}, dim={dim} ===")

    # ------------------------------------------------------------
    # 1. Haar random target from Qiskit
    # ------------------------------------------------------------
    U_target_np = random_unitary(dim).data
    U_target_cp = cp.asarray(U_target_np)

    # Save Qiskit matrix
    np.save("U_target_qiskit.npy", U_target_np)
    print("Saved Qiskit target matrix to U_target_qiskit.npy")

    # ------------------------------------------------------------
    # 2. Identity initialization
    # ------------------------------------------------------------
    U_init = cp.eye(dim, dtype=cp.complex128)

    # ------------------------------------------------------------
    # 3. Run geometric optimizer (your math)
    # ------------------------------------------------------------
    start = time.time()
    U_geo, history = GeometricOptimizer.optimize(
        U_init,
        U_target_cp,
        n_qubits,
        steps=steps,
        frac_ext=0.8,
        lam_reg=0.1,
        step_scale=0.1,
        verbose=True
    )
    end = time.time()

    # Save geometric matrix
    U_geo_np = cp.asnumpy(U_geo)
    np.save("U_geo_matrix.npy", U_geo_np)
    print("Saved geometric matrix to U_geo_matrix.npy")

    # ------------------------------------------------------------
    # 4. Fidelity
    # ------------------------------------------------------------
    fid_op = FidelityObjective.operator_fidelity(U_geo, U_target_cp)

    print(f"\nFinal operator fidelity: {fid_op:.6f}")
    print(f"Total time: {end - start:.3f} seconds")

    return U_geo, U_target_cp


# ------------------------------------------------------------
# 5. Second test: load matrices and compare
# ------------------------------------------------------------
def validate_saved_matrices():
    print("\n=== Matrix Comparison Test ===")

    U_qiskit = np.load("U_target_qiskit.npy")
    U_geo = np.load("U_geo_matrix.npy")

    # Frobenius difference
    diff = np.linalg.norm(U_qiskit - U_geo)
    print(f"Frobenius ||U_qiskit - U_geo|| = {diff:.6e}")

    # Operator fidelity
    overlap = np.trace(U_geo.conj().T @ U_qiskit)
    N = U_qiskit.shape[0]
    fid_op = np.abs(overlap)**2 / (N**2)
    print(f"Operator fidelity: {fid_op:.12f}")

    # Exact equality check
    identical = np.allclose(U_qiskit, U_geo, atol=1e-12)
    print(f"Exact match (allclose): {identical}")


# ------------------------------------------------------------
# 6. Run everything
# ------------------------------------------------------------
if __name__ == "__main__":
    cp.random.seed(0)
    np.random.seed(0)

    # 10 qubits → 1024x1024
    run_large_scale_test(n_qubits=10, steps=600)

    # Compare saved matrices
    validate_saved_matrices()

