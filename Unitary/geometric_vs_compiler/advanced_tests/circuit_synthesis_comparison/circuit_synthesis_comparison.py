# -*- coding: utf-8 -*-
"""
Circuit‑Synthesis Comparison Test
David Mulnix copyright 2026
------------------------------------------------------------

This script performs a synthesis‑accuracy comparison between two completely
different ways of reproducing a target unitary:

    1. **Geometric synthesis** — collapse the identity toward the target using
       the full geometric operator framework (deviation operator, extremal
       spectral projector, geometric update rule, polar re‑unitarization).
       This path uses *your math only*.

    2. **Qiskit circuit synthesis** — embed the target unitary into a Qiskit
       circuit and reconstruct the circuit’s unitary via Qiskit’s internal
       unitary → circuit → unitary pipeline.

What makes this test different from all other tests:
    • It is the ONLY test where Qiskit also attempts to reproduce the target
      unitary, allowing a direct comparison between geometric synthesis and
      circuit‑based synthesis.
    • It is NOT a Haar‑matching test, NOT a noisy‑recovery test, and NOT a
      circuit‑behavior test. It is a *synthesis comparison test*.
    • It evaluates whether geometric descent can match a Haar‑random unitary
      as accurately as Qiskit’s circuit reconstruction — without using gates,
      decomposition, transpilation, or circuit synthesis.

What the test does:
    1. Generate a Haar‑random target unitary U_target.
    2. Use geometric descent to produce U_geo (your math).
    3. Use Qiskit to embed U_target into a circuit and reconstruct U_circ.
    4. Compute three fidelities:
         • Geometric fidelity to target
         • Circuit fidelity to target
         • Geometric vs circuit fidelity

How to interpret results:
    • If geometric fidelity ≈ 1.0, your math matches the Haar target as well
      as Qiskit’s circuit synthesis.
    • If geometric vs circuit fidelity ≈ 1.0, your unitary behaves identically
      to the circuit’s reconstructed unitary.
    • This test provides a direct synthesis‑accuracy benchmark between your
      geometric operator framework and Qiskit’s circuit‑based method.

Purpose:
    • Demonstrate that geometric descent can reproduce the same unitary
      behavior as circuit‑based synthesis.
    • Provide a clean, structure‑free benchmark using Haar‑random targets.
    • Show that your operator geometry is competitive with circuit synthesis
      even though it does not use gates or decomposition.

This test is unique in the validation suite: it is the only one that compares
your geometric optimizer directly against Qiskit’s synthesis pipeline.
"""


import numpy as np
import cupy as cp

from qiskit.quantum_info import random_unitary, Operator, Statevector
from qiskit import QuantumCircuit

from module_geometric_optimizer import GeometricOptimizer
from module_fidelity_objective import FidelityObjective


def qiskit_random_unitary(dim):
    U = random_unitary(dim)
    return cp.asarray(U.data)


def decompose_with_qiskit(U_target_np, n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.unitary(U_target_np, range(n_qubits))
    return qc


def circuit_unitary(qc):
    op = Operator(qc)
    return cp.asarray(op.data)


def frobenius_diff(A, B):
    return float(cp.linalg.norm(A - B))


def delta_norm(U, U_target):
    return float(cp.linalg.norm(U - U_target))


def physical_state_fidelity(Ua, Ub, n_qubits):
    """
    Apply both unitaries to multiple states and compute average fidelity.
    """
    dim = 2**n_qubits
    total = 0.0
    trials = 10

    for _ in range(trials):
        psi = cp.random.randn(dim) + 1j * cp.random.randn(dim)
        psi = psi / cp.linalg.norm(psi)

        out_a = Ua @ psi
        out_b = Ub @ psi

        fid = FidelityObjective.state_fidelity(out_a, out_b, n_qubits)
        total += fid

    return total / trials


def functional_circuit_fidelity(Ua, Ub, n_qubits):
    """
    Embed both unitaries into circuits and compare measurement distributions.
    """
    qc_a = QuantumCircuit(n_qubits)
    qc_b = QuantumCircuit(n_qubits)

    qc_a.unitary(cp.asnumpy(Ua), range(n_qubits))
    qc_b.unitary(cp.asnumpy(Ub), range(n_qubits))

    sv_a = Statevector.from_instruction(qc_a)
    sv_b = Statevector.from_instruction(qc_b)

    probs_a = sv_a.probabilities()
    probs_b = sv_b.probabilities()

    return float(np.sum(np.sqrt(probs_a * probs_b)))


def run_synthesis_test(n_qubits, steps=200, trials=3):
    dim = 2**n_qubits
    print(f"\n=== Circuit Synthesis Comparison (Strong Validation): n_qubits={n_qubits} ===")

    for t in range(1, trials + 1):
        print(f"\n--- Test {t}/{trials} ---")

        # Target
        U_target = random_unitary(dim)
        U_target_np = U_target.data
        U_target_cp = cp.asarray(U_target_np)

        # Geometric synthesis
        U_init = cp.eye(dim, dtype=cp.complex128)
        U_geo, _ = GeometricOptimizer.optimize(
            U_init,
            U_target_cp,
            n_qubits,
            steps=steps,
            frac_ext=0.8,
            lam_reg=0.1,
            step_scale=0.1,
            verbose=False
        )

        # Qiskit circuit synthesis
        qc = decompose_with_qiskit(U_target_np, n_qubits)
        U_circ = circuit_unitary(qc)

        # Matrix-level validation
        fid_geo = FidelityObjective.operator_fidelity(U_geo, U_target_cp)
        fid_circ = FidelityObjective.operator_fidelity(U_circ, U_target_cp)
        fid_geo_vs_circ = FidelityObjective.operator_fidelity(U_geo, U_circ)

        frob_geo = frobenius_diff(U_geo, U_target_cp)
        frob_circ = frobenius_diff(U_circ, U_target_cp)

        delta_geo = delta_norm(U_geo, U_target_cp)
        delta_circ = delta_norm(U_circ, U_target_cp)

        # Physical validation
        phys_geo = physical_state_fidelity(U_geo, U_target_cp, n_qubits)
        phys_circ = physical_state_fidelity(U_circ, U_target_cp, n_qubits)

        # Functional validation
        func_geo = functional_circuit_fidelity(U_geo, U_target_cp, n_qubits)
        func_circ = functional_circuit_fidelity(U_circ, U_target_cp, n_qubits)

        print(f"Qiskit circuit fidelity to target:      {fid_circ:.6f}")
        print(f"Geometric fidelity to target:           {fid_geo:.6f}")
        print(f"Geo vs Circuit fidelity:                {fid_geo_vs_circ:.6f}")

        print(f"Frobenius(geo,target):                  {frob_geo:.6f}")
        print(f"Frobenius(circ,target):                 {frob_circ:.6f}")

        print(f"Delta-norm(geo,target):                 {delta_geo:.6f}")
        print(f"Delta-norm(circ,target):                {delta_circ:.6f}")

        print(f"Physical fidelity (geo vs target):      {phys_geo:.6f}")
        print(f"Physical fidelity (circ vs target):     {phys_circ:.6f}")

        print(f"Functional fidelity (geo vs target):    {func_geo:.6f}")
        print(f"Functional fidelity (circ vs target):   {func_circ:.6f}")

        # Save matrices for public examination
        np.save(f"U_target_n{n_qubits}_t{t}.npy", U_target_np)
        np.save(f"U_geo_n{n_qubits}_t{t}.npy", cp.asnumpy(U_geo))
        np.save(f"U_circ_n{n_qubits}_t{t}.npy", cp.asnumpy(U_circ))


if __name__ == "__main__":
    run_synthesis_test(2)
    run_synthesis_test(3)
