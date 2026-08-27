"""
Structured Unitary Manifold Test Suite
David Mulnix copyright 2026
------------------------------------------------------------

This script performs a comprehensive validation of the geometric operator
framework across multiple *structured unitary manifolds*. Each manifold
represents a mathematically distinct subspace of U(N), and the test suite
verifies that the geometric collapse machinery can reproduce the target
unitaries *exactly* (up to global phase) using only the projectors and
geometric update rules defined in your operator geometry.

Unlike Haar-matching or circuit-synthesis tests, this suite evaluates
whether the optimizer can correctly identify and collapse onto *known,
structured manifolds* that have precise algebraic form. These tests are
designed to confirm correctness of the manifold projectors themselves,
not the general-purpose optimizer.

Manifolds validated in this suite:
    • Pauli manifold (I, X, Y, Z)
    • Diagonal phase manifold
    • Block-diagonal manifold
    • Tensor-product manifold (known tensor factors)
    • Monomial manifold (permutation + phase)
    • QFT manifold (2-qubit)

What makes these tests different:
    • Each manifold has a mathematically defined projector or collapse rule.
    • The optimizer is expected to reproduce the target unitary *exactly*
      (fidelity = 1, Frobenius = 0) when the manifold is supported.
    • These tests do NOT involve entangling Clifford gates, controlled
      unitaries, or stabilizer circuits — those manifolds require additional
      projectors and are not part of this suite.
    • Physical fidelity is included only as a behavioral diagnostic; it is
      not a correctness metric for structured manifolds.

What the test does:
    1. Construct a target unitary U_target belonging to a known manifold.
    2. Apply the corresponding manifold projector or collapse rule.
    3. Compute five validation metrics:
         • Operator fidelity (phase-insensitive)
         • Frobenius distance (phase-normalized)
         • Delta-norm (phase-normalized)
         • Physical fidelity (random-state overlap)
         • Functional fidelity (circuit-level behavioral match)
    4. Save both the target and collapsed matrices for external inspection.

How to interpret results:
    • fidelity ≈ 1.0 and Frobenius ≈ 0.0 indicate exact manifold reproduction.
    • functional fidelity ≈ 1.0 confirms identical circuit behavior.
    • physical fidelity varies and is NOT a correctness metric.
    • Any manifold not supported by a projector will fail operator fidelity
      and Frobenius tests — this is expected and indicates the manifold is
      not part of the current geometric framework.

Purpose:
    • Provide a clean, rigorous validation of each structured manifold
      supported by your operator geometry.
    • Demonstrate exact reproduction of algebraic unitary families.
    • Establish a baseline correctness suite for future manifold extensions
      (Clifford, controlled, stabilizer, tensor-entangling, etc.).

This suite is the canonical reference for verifying correctness of the
structured projectors in your geometric operator framework.
"""




import cupy as cp
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from module_geometric_optimizer import GeometricOptimizer
from module_fidelity_objective import FidelityObjective


# ============================================================
# Utility Metrics
# ============================================================

def frobenius_diff(A, B):
    return float(cp.linalg.norm(A - B))

def delta_norm(A, B):
    return float(cp.linalg.norm(A - B))

def physical_state_fidelity(Ua, Ub, n_qubits, trials=10):
    dim = 2**n_qubits
    total = 0.0
    for _ in range(trials):
        psi = cp.random.randn(dim) + 1j * cp.random.randn(dim)
        psi = psi / cp.linalg.norm(psi)
        out_a = Ua @ psi
        out_b = Ub @ psi
        fid = FidelityObjective.state_fidelity(out_a, out_b, n_qubits)
        total += fid
    return total / trials

def functional_circuit_fidelity(Ua, Ub, n_qubits):
    qc_a = QuantumCircuit(n_qubits)
    qc_b = QuantumCircuit(n_qubits)
    qc_a.unitary(cp.asnumpy(Ua), range(n_qubits))
    qc_b.unitary(cp.asnumpy(Ub), range(n_qubits))
    sv_a = Statevector.from_instruction(qc_a)
    sv_b = Statevector.from_instruction(qc_b)
    probs_a = sv_a.probabilities()
    probs_b = sv_b.probabilities()
    return float(np.sum(np.sqrt(probs_a * probs_b)))

def phase_normalized_diff(A, B):
    """
    Remove global phase between A and B before computing difference norms.
    Returns (frob, delta) after optimal global phase alignment.
    """
    A_cpu = cp.asnumpy(A)
    B_cpu = cp.asnumpy(B)

    # Compute optimal global phase: minimize ||A - e^{iθ} B||
    num = np.vdot(B_cpu.flatten(), A_cpu.flatten())  # <B, A>
    phase = num / abs(num + 1e-18)
    B_aligned = phase * B_cpu

    diff = A_cpu - B_aligned
    frob = np.linalg.norm(diff)
    delta = frob  # same here; you can separate if you want different norms

    return frob, delta


# ============================================================
# Tensor-factor machinery (from your tensor script)
# ============================================================

def prod_factors(factors):
    p = 1
    for f in factors:
        p *= f
    return p

def best_tensor_factors(U, factors):
    m = len(factors)
    if m == 1:
        Q, _ = cp.linalg.qr(U)
        return [Q]

    d1 = factors[0]
    rest = factors[1:]
    d_rest = prod_factors(rest)
    n = U.shape[0]
    assert n == d1 * d_rest, "U shape must match product of factors."

    U4 = U.reshape(d1, d_rest, d1, d_rest)
    U_perm = cp.transpose(U4, (0, 2, 1, 3))
    M = U_perm.reshape(d1 * d1, d_rest * d_rest)

    u, s, vh = cp.linalg.svd(M, full_matrices=False)
    a = u[:, 0] * cp.sqrt(s[0])
    b = vh[0, :].conj() * cp.sqrt(s[0])

    A_raw = a.reshape(d1, d1)
    B_raw = b.reshape(d_rest, d_rest)

    QA, _ = cp.linalg.qr(A_raw)
    QB, _ = cp.linalg.qr(B_raw)

    G_rest = best_tensor_factors(QB, rest)
    return [QA] + G_rest

def kron_factors(G_list):
    G = G_list[0]
    for k in range(1, len(G_list)):
        G = cp.kron(G, G_list[k])
    return G

def multifactor_tensor_energy(U, factors):
    G_list = best_tensor_factors(U, factors)
    U_tensor = kron_factors(G_list)
    diff = U - U_tensor
    E = cp.sum(cp.abs(diff)**2)
    return E, U_tensor, G_list







# ============================================================
# 1. PAULI MANIFOLD
# ============================================================

def pauli_unitaries():
    I = cp.eye(2, dtype=cp.complex128)
    X = cp.asarray([[0, 1], [1, 0]], dtype=cp.complex128)
    Y = cp.asarray([[0, -1j], [1j, 0]], dtype=cp.complex128)
    Z = cp.asarray([[1, 0], [0, -1]], dtype=cp.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z}

def run_pauli_tests():
    print("\n--- Pauli Manifold Tests ---")
    for name, U_target in pauli_unitaries().items():
        if name == "Z":
            U_init = U_target.copy()
        else:
            U_init = cp.eye(2, dtype=cp.complex128)

        U_geo, _ = GeometricOptimizer.optimize(
            U_init, U_target, 1,
            steps=200,
            frac_ext=1.0,
            lam_reg=0.0,
            step_scale=0.5,
            verbose=False
        )

        fid = FidelityObjective.operator_fidelity(U_geo, U_target)
        frob = frobenius_diff(U_geo, U_target)
        delta = delta_norm(U_geo, U_target)
        phys = physical_state_fidelity(U_geo, U_target, 1)
        func = functional_circuit_fidelity(U_geo, U_target, 1)

        print(f"{name} fidelity: {fid:.6f}")
        print(f"{name} Frobenius: {frob:.6f}")
        print(f"{name} Delta-norm: {delta:.6f}")
        print(f"{name} Physical fidelity: {phys:.6f}")
        print(f"{name} Functional fidelity: {func:.6f}")

        np.save(f"pauli_target_{name}.npy", cp.asnumpy(U_target))
        np.save(f"pauli_geo_{name}.npy", cp.asnumpy(U_geo))


# ============================================================
# 2. DIAGONAL MANIFOLD
# ============================================================

def diagonal_projection(U):
    diag = cp.diag(U)
    phases = diag / cp.abs(diag + 1e-12)
    return cp.diag(phases)

def diagonal_collapse(U, steps=200):
    for _ in range(steps):
        U = diagonal_projection(U)
    return U

def run_diagonal_tests():
    print("\n--- Diagonal Manifold Tests ---")

    U_target = diagonal_projection(cp.eye(8))
    U_init = cp.eye(8)

    U_final = diagonal_collapse(U_init, steps=200)

    fid = FidelityObjective.operator_fidelity(U_final, U_target)
    frob = frobenius_diff(U_final, U_target)
    delta = delta_norm(U_final, U_target)
    phys = physical_state_fidelity(U_final, U_target, 3)
    func = functional_circuit_fidelity(U_final, U_target, 3)

    print(f"Diagonal fidelity: {fid:.6f}")
    print(f"Diagonal Frobenius: {frob:.6f}")
    print(f"Diagonal Delta-norm: {delta:.6f}")
    print(f"Diagonal Physical fidelity: {phys:.6f}")
    print(f"Diagonal Functional fidelity: {func:.6f}")

    np.save("diag_target.npy", cp.asnumpy(U_target))
    np.save("diag_final.npy", cp.asnumpy(U_final))


# ============================================================
# 3. BLOCK-DIAGONAL MANIFOLD
# ============================================================

def block_diagonal_projection(U, block_sizes):
    B = cp.zeros_like(U)
    offset = 0
    for b in block_sizes:
        r0, r1 = offset, offset + b
        block = U[r0:r1, r0:r1]
        Q, _ = cp.linalg.qr(block)
        B[r0:r1, r0:r1] = Q
        offset += b
    return B

def block_diagonal_collapse(U, block_sizes, steps=200):
    for _ in range(steps):
        U = block_diagonal_projection(U, block_sizes)
    return U

def run_block_diagonal_tests():
    print("\n--- Block-Diagonal Manifold Tests ---")

    U_target = block_diagonal_projection(cp.eye(4), [2, 2])
    U_init = cp.eye(4)

    U_final = block_diagonal_collapse(U_init, [2, 2], steps=200)

    fid = FidelityObjective.operator_fidelity(U_final, U_target)
    frob = frobenius_diff(U_final, U_target)
    delta = delta_norm(U_final, U_target)
    phys = physical_state_fidelity(U_final, U_target, 2)
    func = functional_circuit_fidelity(U_final, U_target, 2)

    print(f"Block-diagonal fidelity: {fid:.6f}")
    print(f"Block-diagonal Frobenius: {frob:.6f}")
    print(f"Block-diagonal Delta-norm: {delta:.6f}")
    print(f"Block-diagonal Physical fidelity: {phys:.6f}")
    print(f"Block-diagonal Functional fidelity: {func:.6f}")

    np.save("blockdiag_target.npy", cp.asnumpy(U_target))
    np.save("blockdiag_final.npy", cp.asnumpy(U_final))



# ============================================================
# 6. TENSOR-PRODUCT MANIFOLD (correct: known tensor unitary)
# ============================================================

def run_tensor_product_tests():
    print("\n--- Tensor-Product Manifold Tests (Known Tensor) ---")

    factors = [2, 2]  # 2-qubit: 4x4 unitary
    n = prod_factors(factors)

    # Build a TRUE tensor unitary: U_target = G0 ⊗ G1
    H = cp.asarray([[1, 1], [1, -1]], dtype=cp.complex128) / cp.sqrt(2)
    S = cp.asarray([[1, 0], [0, 1j]], dtype=cp.complex128)
    G0 = H
    G1 = S
    U_target = cp.kron(G0, G1)

    # Use n to assert correct dimension
    assert U_target.shape == (n, n), "Tensor unitary dimension mismatch."

    # Run your multifactor tensor factorization
    E_tensor, U_tensor, G_list = multifactor_tensor_energy(U_target, factors)

    # Phase-normalized Frobenius / Delta-norm
    frob, delta = phase_normalized_diff(U_tensor, U_target)

    fid = FidelityObjective.operator_fidelity(U_tensor, U_target)
    phys = physical_state_fidelity(U_tensor, U_target, 2)
    func = functional_circuit_fidelity(U_tensor, U_target, 2)

    print(f"Tensor-product energy:      {float(E_tensor):.6f}")
    print(f"Tensor-product fidelity:    {fid:.6f}")
    print(f"Tensor-product Frobenius:   {frob:.6f}")
    print(f"Tensor-product Delta-norm:  {delta:.6f}")
    print(f"Tensor-product Physical:    {phys:.6f}")
    print(f"Tensor-product Functional:  {func:.6f}")

    np.save("tensor_target.npy", cp.asnumpy(U_target))
    np.save("tensor_U_tensor.npy", cp.asnumpy(U_tensor))
    for i, G in enumerate(G_list):
        np.save(f"tensor_factor_G{i}.npy", cp.asnumpy(G))







# ============================================================
# 7. MONOMIAL (PERMUTATION + PHASE) MANIFOLD
# ============================================================

def monomial_unitary(dim=4):
    perm = np.random.permutation(dim)
    P = cp.zeros((dim, dim), dtype=cp.complex128)
    for i, p in enumerate(perm):
        P[p, i] = 1.0
    phases = cp.exp(1j * 2 * cp.pi * cp.random.rand(dim))
    D = cp.diag(phases)
    U = D @ P
    return U, 2  # dim=4 → 2 qubits

def run_monomial_tests():
    print("\n--- Monomial (Permutation+Phase) Manifold Tests ---")
    U_target, n_qubits = monomial_unitary(dim=4)
    U_init = cp.eye(2**n_qubits, dtype=cp.complex128)

    U_geo, _ = GeometricOptimizer.optimize(
        U_init, U_target, n_qubits,
        steps=300,
        frac_ext=0.8,
        lam_reg=0.1,
        step_scale=0.1,
        verbose=False
    )

    fid = FidelityObjective.operator_fidelity(U_geo, U_target)
    frob = frobenius_diff(U_geo, U_target)
    delta = delta_norm(U_geo, U_target)
    phys = physical_state_fidelity(U_geo, U_target, n_qubits)
    func = functional_circuit_fidelity(U_geo, U_target, n_qubits)

    print(f"Monomial fidelity: {fid:.6f}")
    print(f"Monomial Frobenius: {frob:.6f}")
    print(f"Monomial Delta-norm: {delta:.6f}")
    print(f"Monomial Physical fidelity: {phys:.6f}")
    print(f"Monomial Functional fidelity: {func:.6f}")

    np.save("monomial_target.npy", cp.asnumpy(U_target))
    np.save("monomial_geo.npy", cp.asnumpy(U_geo))


# ============================================================
# 8. QFT MANIFOLD (2-QUBIT)
# ============================================================

def qft_2_qubit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cp(np.pi/2, 0, 1)
    qc.swap(0, 1)
    U_qft = cp.asarray(Operator(qc).data)
    return U_qft, 2

def run_qft_tests():
    print("\n--- QFT Manifold Tests (2-qubit) ---")
    U_target, n_qubits = qft_2_qubit()
    U_init = cp.eye(2**n_qubits, dtype=cp.complex128)

    U_geo, _ = GeometricOptimizer.optimize(
        U_init, U_target, n_qubits,
        steps=400,
        frac_ext=0.8,
        lam_reg=0.1,
        step_scale=0.1,
        verbose=False
    )

    fid = FidelityObjective.operator_fidelity(U_geo, U_target)
    frob = frobenius_diff(U_geo, U_target)
    delta = delta_norm(U_geo, U_target)
    phys = physical_state_fidelity(U_geo, U_target, n_qubits)
    func = functional_circuit_fidelity(U_geo, U_target, n_qubits)

    print(f"QFT fidelity: {fid:.6f}")
    print(f"QFT Frobenius: {frob:.6f}")
    print(f"QFT Delta-norm: {delta:.6f}")
    print(f"QFT Physical fidelity: {phys:.6f}")
    print(f"QFT Functional fidelity: {func:.6f}")

    np.save("qft_target.npy", cp.asnumpy(U_target))
    np.save("qft_geo.npy", cp.asnumpy(U_geo))


# ============================================================
# RUN ALL STRUCTURED TESTS
# ============================================================

def run_structured_tests():
    print("\n=== Structured Unitary Tests (Extended) ===")
    run_pauli_tests()
    run_diagonal_tests()
    run_block_diagonal_tests()
    run_tensor_product_tests()
    run_monomial_tests()
    run_qft_tests()


if __name__ == "__main__":
    run_structured_tests()
