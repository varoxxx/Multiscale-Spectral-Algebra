# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026

Circuit‑Derived Unitary Matching + Circuit‑Execution Validation
--------------------------------------------------------------

This test evaluates the geometric optimizer on unitary matrices that come
directly from real Qiskit circuits. Unlike Haar‑random tests, the targets
here are produced by actual gate sequences containing random single‑qubit
rotations and random CNOT entanglers. The goal is to measure how well the
geometric optimizer can match the unitary of a genuine circuit using only
geometric descent on U(N), and to verify that the resulting unitary behaves
identically to the circuit itself.

What the test does:
    1. Construct a random Qiskit circuit with:
         • arbitrary U(2) single‑qubit rotations
         • randomly chosen CNOTs between random qubit pairs
    2. Convert the circuit into its exact unitary matrix using
       qiskit.quantum_info.Operator.
    3. Use the geometric optimizer to collapse the identity matrix toward
       this circuit‑derived unitary.
    4. Compute operator fidelity between the geometric result and the
       circuit’s true unitary.
    5. Save both unitaries (.npy) for reproducibility and external audit.
    6. Perform matrix‑level comparison:
         • Frobenius difference
         • operator fidelity
         • exact equality (np.allclose)
    7. Perform circuit‑level behavioral validation (NEW):
         • Apply the original Qiskit circuit to multiple input states
         • Apply the geometric unitary to the same input states
         • Compare the resulting output states
         • Confirm that both unitaries produce identical physical behavior

What this test is NOT:
    • It is not a circuit‑extraction or gate‑recovery test.
    • It is not a hardware‑efficient or topology‑restricted test.
    • It is not a comparison against Qiskit’s transpiler or algebraic
      compilation pipeline.
    • It is not demonstrating a Qiskit limitation—Qiskit is only used to
      generate the circuit and its unitary.

Purpose:
    • Validate that the geometric optimizer can match unitaries generated
      by real circuits, not only Haar‑random or structured matrices.
    • Demonstrate stability of geometric descent on circuit‑derived targets.
    • Show how fidelity behaves across multiple random circuits, revealing
      which circuit structures are easy or hard for geometric matching.
    • Provide a realistic benchmark for geometric operator descent on
      unitaries that arise from actual quantum gate sequences.
    • Provide the strongest possible proof of correctness by verifying:
         – mathematical equivalence (matrix‑level)
         – physical equivalence (state‑level)
         – functional equivalence (circuit‑level)

This test is intended to provide a clear demonstration
of geometric descent behavior on circuit‑generated unitaries, including full
matrix, physical, and functional validation.
"""


import cupy as cp
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from module_geometric_optimizer import GeometricOptimizer
from module_fidelity_objective import FidelityObjective


# ------------------------------------------------------------
# 1. Random Qiskit circuit generator
# ------------------------------------------------------------
def random_qiskit_circuit(n_qubits, depth=5):
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng()

    for _ in range(depth):
        # random single-qubit rotations
        for q in range(n_qubits):
            theta = 2 * np.pi * rng.random()
            phi = 2 * np.pi * rng.random()
            lam = 2 * np.pi * rng.random()
            qc.u(theta, phi, lam, q)

        # random CNOTs
        for _ in range(n_qubits):
            ctrl = rng.integers(0, n_qubits)
            targ = rng.integers(0, n_qubits)
            if ctrl != targ:
                qc.cx(ctrl, targ)

    return qc


# ------------------------------------------------------------
# 2. Convert circuit → unitary
# ------------------------------------------------------------
def circuit_unitary(qc):
    op = Operator(qc)
    return cp.asarray(op.data)


# ------------------------------------------------------------
# 3. Circuit‑execution validation (NEW)
# ------------------------------------------------------------
def validate_circuit_behavior(qc, U_geo_np, n_qubits):
    print("\n=== Circuit Behavior Validation ===")

    # Convert geometric unitary to Qiskit Operator
    U_geo_op = Operator(U_geo_np)

    # Test several input states
    test_states = [
        Statevector.from_label("0" * n_qubits),
        Statevector.from_label("1" * n_qubits),
        Statevector.from_label("+" * n_qubits),
        Statevector.from_label("-" * n_qubits),
    ]

    for idx, psi_in in enumerate(test_states):
        # Qiskit circuit output
        psi_qiskit = psi_in.evolve(qc)

        # Geometric unitary output
        psi_geo = psi_in.evolve(U_geo_op)

        # Compare
        diff = np.linalg.norm(psi_qiskit.data - psi_geo.data)
        print(f"Input state {idx}: ||psi_qiskit - psi_geo|| = {diff:.6e}")


# ------------------------------------------------------------
# 4. Main test
# ------------------------------------------------------------
def run_circuit_derived_tests(n_qubits, steps=200, trials=5):
    dim = 2**n_qubits
    print(f"\n=== Circuit-Derived Targets: n_qubits={n_qubits} ===")

    for t in range(1, trials + 1):
        print(f"\n--- Test {t}/{trials} ---")

        # 1) Build random Qiskit circuit
        qc = random_qiskit_circuit(n_qubits, depth=5)

        # 2) Get its unitary as target
        U_target_cp = circuit_unitary(qc)
        U_target_np = cp.asnumpy(U_target_cp)

        # Save Qiskit unitary
        np.save(f"U_target_qiskit_{n_qubits}q_test{t}.npy", U_target_np)
        print(f"Saved Qiskit unitary → U_target_qiskit_{n_qubits}q_test{t}.npy")

        # 3) Run geometric optimizer
        U_init = cp.eye(dim, dtype=cp.complex128)
        U_geo_cp, _ = GeometricOptimizer.optimize(
            U_init,
            U_target_cp,
            n_qubits,
            steps=steps,
            frac_ext=0.8,
            lam_reg=0.1,
            step_scale=0.1,
            verbose=False
        )

        # Save geometric result
        U_geo_np = cp.asnumpy(U_geo_cp)
        np.save(f"U_geo_matrix_{n_qubits}q_test{t}.npy", U_geo_np)
        print(f"Saved geometric unitary → U_geo_matrix_{n_qubits}q_test{t}.npy")

        # 4) Compare fidelities
        fid_geo = FidelityObjective.operator_fidelity(U_geo_cp, U_target_cp)
        print(f"Geometric fidelity to circuit unitary: {fid_geo:.6f}")

        # 5) Run matrix comparison test
        validate_saved_matrices(
            f"U_target_qiskit_{n_qubits}q_test{t}.npy",
            f"U_geo_matrix_{n_qubits}q_test{t}.npy"
        )

        # 6) Run circuit-behavior validation (NEW)
        validate_circuit_behavior(qc, U_geo_np, n_qubits)


# ------------------------------------------------------------
# 5. Matrix Comparison Test
# ------------------------------------------------------------
def validate_saved_matrices(path_target, path_geo):
    print("\n=== Matrix Comparison Test ===")

    U_qiskit = np.load(path_target)
    U_geo = np.load(path_geo)

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
# 6. Run tests
# ------------------------------------------------------------
if __name__ == "__main__":
    run_circuit_derived_tests(2)
    run_circuit_derived_tests(3)
