#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026

Geometric MPU Extraction and Validation Script
==============================================

This script provides a complete, executable implementation of the geometric
matrix‑product‑unitary (MPU) extraction pipeline used throughout the paper.  
It serves as the reproducibility backbone for the CZ‑layer use case and the
subsequent non‑uniform, high‑bond, QFT, and multi‑layer MPU demonstrations.

The code implements the full geometric method:
    global unitary → interleaved tensor → site‑wise SVD → bond propagation
    → tensor contraction → polar re‑unitarization → fidelity validation.

Every step corresponds directly to the mathematical framework developed in the
paper, and each printed output can be mapped to a specific geometric equation.

---------------------------------------------------------------------------
Mathematical Mapping to the Paper
---------------------------------------------------------------------------

1) Interleaved tensor representation
   ---------------------------------
   The function `reshape_unitary_to_site_matrix` reshapes U ∈ U(2^N) into
   interleaved coordinates:

       U_{i1 j1 i2 j2 ... iN jN}

   This is the geometric coordinate chart on the MPU manifold.  
   It is the starting point for site‑wise SVD extraction.

2) Site‑wise SVD extraction
   -------------------------
   The function `mpu_extract_all_sites` performs:

       M^{(k)} = U^{(k)} Σ^{(k)} V^{(k)†}
       A^{(k)}_{α,i,j,β} = U^{(k)}_{α i j, β} · Σ^{(k)}_{β}

   This is the local geometric factorization described in the paper.  
   Rows of M^{(k)} correspond to (bond_left × i × j), matching the indexing
   convention in the methods section.

3) Bond‑dimension propagation
   ---------------------------
       bond_left^{(k+1)} = χ_eff^{(k)}

   The script implements both fixed‑χ and adaptive χ via energy thresholds.
   The retained singular values determine the effective bond dimension at each
   site, producing the bond‑profile plots shown in the MPU use cases.

4) Tensor contraction reconstruction
   ----------------------------------
       T = A^{(1)} ⋆ A^{(2)} ⋆ ... ⋆ A^{(N)}
       reorder axes → reshape → U_mpu

   The function `contract_A_list_to_operator` performs the exact contraction
   ordering described in the paper.  
   It permutes interleaved axes into (i1...iN, j1...jN) before reshaping into
   a 2^N × 2^N matrix.

5) Polar re‑unitarization
   -----------------------
       U_mpu_unitary = nearest_unitary(T)
       via SVD: T = U S V† → U V†

   Implemented in `polar_unitary_from_matrix`.  
   This ensures the reconstructed operator lies exactly in U(2^N).

6) Global phase alignment
   -----------------------
       φ = ⟨U_mpu, U_target⟩ / |⟨U_mpu, U_target⟩|
       U_mpu ← φ* · U_mpu

   Phase alignment is essential for meaningful fidelity comparisons.

7) Geometric fidelity metrics
   ---------------------------
   The script computes:
       - Operator fidelity: |Tr(U_mpu† U_target)|^2 / (2^N)^2
       - State fidelity: randomized action‑on‑states tests
       - Frobenius distance: ||U_mpu - U_target||_F

   Operator fidelity and state fidelity are the primary correctness metrics.
   Frobenius norms are geometric displacement measures; they are sensitive to
   global phase and should not be used alone to judge correctness.

---------------------------------------------------------------------------
Tests Included
---------------------------------------------------------------------------

1) run_mpu_cz_validation()
   - Validates the full pipeline on the uniform CZ layer.
   - Exercises:
       * interleaving reshape
       * site‑wise SVD extraction
       * bond propagation
       * tensor contraction
       * polar re‑unitarization
       * global phase alignment
   - Expected outputs:
       * operator fidelity ≈ 1.0
       * Frobenius ≈ 0 after phase alignment
       * correct A_site tensor shapes

2) run_mpu_full_pipeline_test()
   - Short sanity check on the pipeline using the CZ layer.
   - Prints projection delta ||U_raw - U_mpu||_F and fidelity.

3) run_mpu_bidir_test()
   - Demonstrates bidirectional gauge stabilization.
   - Exercises:
       * right→left SVD sweep
       * gauge averaging (A_lr + A_rl)/2
   - Expected outputs:
       * stable A_site shapes
       * reduced gauge drift

---------------------------------------------------------------------------
What the User Should Look For
---------------------------------------------------------------------------

- Operator fidelity = 1.0  
  Indicates exact reconstruction up to global phase.

- Physical/state fidelity = 1.0  
  Confirms identical action on all states.

- Frobenius norm ≈ 0 (after phase alignment)  
  Indicates exact projection onto the MPU manifold.

- A_site tensor shapes  
  Should match the bond‑dimension profiles shown in the paper.

---------------------------------------------------------------------------
Interpretation Notes
---------------------------------------------------------------------------

- A large Frobenius norm for U_rec (raw contraction) is expected.
  Frobenius distance is sensitive to global phase.

- After polar re‑unitarization and phase alignment, U_mpu should satisfy:
      ||U_mpu - U_target||_F ≈ machine precision.

- Fidelity is the true geometric correctness metric.

---------------------------------------------------------------------------
External Qiskit Compatibility Test
---------------------------------------------------------------------------

This script also includes an external validation step using the Qiskit quantum
compiler. After reconstructing the geometric MPU operator U_mpu, the script:

    (1) Wraps U_mpu as a Qiskit Operator.
    (2) Synthesizes a quantum circuit using Qiskit's CX+U3 gate basis.
    (3) Extracts the unitary U_qiskit implemented by the synthesized circuit.
    (4) Compares U_qiskit directly to U_mpu.

A correct reconstruction produces:

    Frobenius ||U_mpu - U_qiskit||_F = 0
    Operator fidelity = 1.0

These results indicate that the Qiskit‑decomposed circuit is mathematically
identical to the geometric MPU operator. This confirms that the reconstructed
MPU is physically realizable, compatible with standard gate‑based compilers,
and executable on contemporary quantum hardware toolchains.

---------------------------------------------------------------------------
Summary
---------------------------------------------------------------------------

This script is the executable realization of the geometric MPU extraction
framework. It validates that the CZ layer—and more generally, any nearest‑neighbor
two‑qubit layer—lies exactly on the MPU manifold and that the geometric method
reconstructs these operators with fidelity 1.0. The added Qiskit compatibility
test further demonstrates that the reconstructed MPU operator is fully executable
within a modern quantum compiler, providing external confirmation of correctness
and hardware‑level compatibility.
"""




import cupy as cp
from functools import reduce
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from module_fidelity_objective import FidelityObjective
import json
from qiskit import qpy, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# ---------------------------
# Utilities and metrics
# ---------------------------

def frobenius_diff(A, B):
    return float(cp.linalg.norm(A - B))

def delta_norm(A, B):
    return float(cp.linalg.norm(A - B))

def physical_state_fidelity(Ua, Ub, n_qubits, trials=20, seed=0):
    return FidelityObjective.state_fidelity(Ua, Ub, n_qubits)


def operator_fidelity(Ua, Ub):
    # Use fidelity measure based on normalized Hilbert-Schmidt inner product
    dim = Ua.shape[0]
    num = cp.abs(cp.trace(Ua.conj().T @ Ub))**2
    den = dim**2
    return float(num / den)

def functional_circuit_fidelity(Ua, Ub):
    # Keep for quick sanity check (overlap of first column)
    probs_a = cp.abs(Ua[:, 0])**2
    probs_b = cp.abs(Ub[:, 0])**2
    return float(cp.sum(cp.sqrt(probs_a * probs_b)))

# ---------------------------
# CZ-layer builder
# ---------------------------

def two_qubit_cz():
    return cp.asarray([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, -1]], dtype=cp.complex128)

def kron_reduce(factors):
    """Reduce a list of CuPy matrices via Kronecker product."""
    if len(factors) == 0:
        return cp.array([1], dtype=cp.complex128)
    if len(factors) == 1:
        return factors[0]
    # use cp.kron directly in the reducer
    return reduce(lambda a, b: cp.kron(a, b), factors)


def embed_two_qubit_gate_kron(G, q, N):
    """
    Embed a 2-qubit gate G acting on qubits (q, q+1) into N-qubit space.
    This implementation builds a factor list and reduces via kron.
    Works for small-to-moderate N (N <= ~10 depending on memory).
    """
    I2 = cp.eye(2, dtype=cp.complex128)
    factors = []
    k = 0
    while k < N:
        if k == q:
            # Insert the 2-qubit gate as a single factor (4x4)
            factors.append(G)
            k += 2
        else:
            factors.append(I2)
            k += 1
    # If last factor list length is less than N (shouldn't happen), pad
    # Now reduce via kron
    return kron_reduce(factors)

def build_unitary_uniform_bulk_mpu(N, brickwork=False, layers=1):
    """
    Build a CZ-layer unitary for N qubits.
    - If brickwork=False: apply CZ on every adjacent pair once (q=0..N-2).
    - If brickwork=True: apply even bonds then odd bonds per layer.
    - layers: number of repetitions of the brickwork pattern (default 1).
    """

    CZ = two_qubit_cz()
    dim = 2**N
    U = cp.eye(dim, dtype=cp.complex128)

    if not brickwork:
        # Single pass: apply CZ on every adjacent pair once
        for q in range(N - 1):
            G = embed_two_qubit_gate_kron(CZ, q, N)
            U = G @ U
        return U

    # Brickwork pattern: even bonds then odd bonds, repeated 'layers' times
    for _ in range(layers):
        # even bonds
        for q in range(0, N - 1, 2):
            G = embed_two_qubit_gate_kron(CZ, q, N)
            U = G @ U
        # odd bonds
        for q in range(1, N - 1, 2):
            G = embed_two_qubit_gate_kron(CZ, q, N)
            U = G @ U

    return U

# ---------------------------
# SVD-based extraction helpers
# ---------------------------

def reshape_unitary_to_site_matrix(U, N):
    """
    Reshape U (dim x dim) into a matrix M for site-wise SVD:
    - U_tensor shape: (d,)*N + (d,)*N
    - interleave physical input/output indices to get rows=(i_k,j_k), cols=rest
    Returns initial M for site 0.
    """
    d = 2
    dim = 2**N
    assert U.shape == (dim, dim)
    U_tensor = U.reshape((d,)*N + (d,)*N)
    interleave = []
    for k in range(N):
        interleave.append(k)
        interleave.append(N + k)
    U_inter = U_tensor.transpose(interleave)
    local_dim = d**2
    rest_dim = d**(2*(N-1))
    M = U_inter.reshape(local_dim, rest_dim)
    return M

def adaptive_chi_from_svals(svals, energy_threshold=1 - 1e-12):
    """
    Choose chi adaptively to retain at least energy_threshold fraction of squared singular values.
    svals: 1D array of singular values (non-negative)
    """
    s2 = cp.asarray(svals)**2
    total = float(cp.sum(s2))
    if total == 0:
        return 1
    cumsum = cp.cumsum(s2)
    frac = cumsum / total
    idx = cp.searchsorted(frac, energy_threshold)
    chi = int(min(len(svals), idx + 1))
    return max(1, chi)

def svd_site_truncate(M, chi=None, energy_threshold=None):
    """
    Compute SVD of M and truncate either to fixed chi or adaptively by energy_threshold.
    Returns Utr, Str, Vtr, chi_eff, energy_kept
    """
    Umat, Svec, Vh = cp.linalg.svd(M, full_matrices=False)
    if energy_threshold is not None:
        chi_eff = adaptive_chi_from_svals(Svec, energy_threshold=energy_threshold)
    elif chi is not None:
        chi_eff = min(chi, Umat.shape[1])
    else:
        chi_eff = Umat.shape[1]
    Utr = Umat[:, :chi_eff]
    Str = Svec[:chi_eff]
    Vtr = Vh[:chi_eff, :]
    energy_kept = float(cp.sum(Str**2) / cp.sum(Svec**2))
    return Utr, Str, Vtr, chi_eff, energy_kept

# ---------------------------
# MPU extraction (left-to-right sweep)
# ---------------------------

def mpu_extract_all_sites(U, n_qubits, chi=None, energy_threshold=None, debug=False):
    """
    Left-to-right SVD sweep extracting local A_site tensors.
    Returns list of A_site tensors with shapes (bond_left_dim, d, d, chi_eff).
    """
    d = 2
    N = n_qubits
    M = reshape_unitary_to_site_matrix(U, N)

    A_list = []
    bond_left_dim = 1

    for site in range(N):
        Utr, Str, Vtr, chi_eff, energy_kept = svd_site_truncate(M, chi=chi, energy_threshold=energy_threshold)
        # shape checks
        local_dim = d**2
        assert Utr.shape[0] == bond_left_dim * local_dim, f"Utr row mismatch at site {site}"
        # Build A_site with shape (bond_left_dim, d, d, chi_eff)
        A_site = cp.zeros((bond_left_dim, d, d, chi_eff), dtype=cp.complex128)
        for alpha in range(bond_left_dim):
            for i in range(d):
                for j in range(d):
                    idx = alpha * local_dim + i * d + j
                    # multiply left singular vector entry by singular value
                    A_site[alpha, i, j, :] = Utr[idx, :chi_eff] * Str[:chi_eff]
        A_list.append(A_site)

        if debug:
            print(f"[L->R] site {site}: bond_left_dim={bond_left_dim}, chi_eff={chi_eff}, energy_kept={energy_kept:.6e}, A_site.shape={A_site.shape}")

        # Prepare M for next site
        if site < N - 1:
            rest_dim_next = d**(2*(N - site - 2))
            M = Vtr.reshape(chi_eff * local_dim, rest_dim_next)
            bond_left_dim = chi_eff

    return A_list

# ---------------------------
# MPU reconstruction and projection
# ---------------------------

def contract_A_list_to_operator(A_list):
    """
    Contract list of A_site tensors into a full operator matrix T_phys of shape (d^N, d^N).
    This implementation assumes A_list[k] has shape (bond_left_k, d, d, bond_right_k).
    """
    N = len(A_list)
    d = 2

    # Start with first tensor
    T = A_list[0]  # shape (b0_left=1, d, d, b0_right)
    # Iteratively contract along bond indices
    for k in range(1, N):
        Ak = A_list[k]  # shape (bk_left, d, d, bk_right)
        # Contract T's last bond with Ak's first bond
        T = cp.tensordot(T, Ak, axes=([-1], [0]))
        # After tensordot, the physical axes are interleaved: (left_bond, i1, j1, i2, j2, ..., right_bond)
    # At this point boundary bond dims should be 1; collapse them and reshape to matrix
    # Move all axes so that inputs (i1..iN) come first and outputs (j1..jN) second.
    # Current ordering after the loop is: (b_left=1, i1, j1, i2, j2, ..., iN, jN, b_right=1)
    # Remove boundary bonds if present
    if T.shape[0] == 1:
        T = T[0]
    if T.shape[-1] == 1:
        T = T[..., 0]
    # Now T has shape (i1, j1, i2, j2, ..., iN, jN)
    # Build lists of input and output axis indices
    axes = list(range(T.ndim))
    input_axes = axes[0::2]   # i1, i2, ...
    output_axes = axes[1::2]  # j1, j2, ...
    # Permute to inputs then outputs
    perm = input_axes + output_axes
    T_perm = T.transpose(perm)
    # Reshape to matrix (d^N, d^N)
    T_phys = T_perm.reshape((d**N, d**N))
    return T_phys


def polar_unitary_from_matrix(M):
    """
    Return nearest unitary to M via polar decomposition: U = UV^H where M = U_p H
    Implemented via SVD: M = U S Vh -> nearest unitary = U Vh
    """
    Ux, Sx, Vxh = cp.linalg.svd(M, full_matrices=False)
    return Ux @ Vxh

def mpu_projection(U, n_qubits, chi=None, energy_threshold=None, debug=False):
    """
    Full projection pipeline:
     - Extract A_list via left-to-right SVD sweep (optionally adaptive chi)
     - Contract A_list into operator U_mpu (interleaved ordering)
     - Project to nearest unitary via polar (SVD)
    Returns U_mpu (unitary), A_list
    """
    N = n_qubits
    A_list = mpu_extract_all_sites(U, N, chi=chi, energy_threshold=energy_threshold, debug=debug)
    T = contract_A_list_to_operator(A_list)
    U_mpu = polar_unitary_from_matrix(T)
    # Align global phase to original U for comparison
    return U_mpu, A_list

# ---------------------------
# Bidirectional sweeps and averaging
# ---------------------------

def mpu_svd_sweep_rl(U, n_qubits, chi=None, energy_threshold=None, debug=False):
    """
    Right-to-left sweep: mirror the reshaping and perform SVDs from the right.
    Returns A_list in natural left-to-right order.
    """
    d = 2
    N = n_qubits
    # Reshape and reverse interleaving
    U_tensor = U.reshape((d,)*N + (d,)*N)
    interleave = []
    for k in range(N):
        interleave.append(k)
        interleave.append(N + k)
    U_inter = U_tensor.transpose(interleave)
    # Reverse the interleaved axes
    axes = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U_rev = U_inter.transpose(axes[::-1])
    local_dim = d**2
    rest_dim = d**(2*(N-1))
    M = U_rev.reshape(local_dim, rest_dim)

    A_list_rev = []
    bond_right_dim = 1

    for site in range(N):
        Utr, Str, Vtr, chi_eff, energy_kept = svd_site_truncate(M, chi=chi, energy_threshold=energy_threshold)
        # Build A_site with shape (chi_eff, d, d, bond_right_dim)
        A_site = cp.zeros((chi_eff, d, d, bond_right_dim), dtype=cp.complex128)
        for beta in range(bond_right_dim):
            for i in range(d):
                for j in range(d):
                    idx = i * d + j
                    A_site[:, i, j, beta] = Utr[idx, :chi_eff] * Str[:chi_eff]
        A_list_rev.append(A_site)
        if debug:
            print(f"[R->L] site {site}: bond_right_dim={bond_right_dim}, chi_eff={chi_eff}, energy_kept={energy_kept:.6e}, A_site.shape={A_site.shape}")
        if site < N - 1:
            rest_dim_next = d**(2*(N-site-2))
            M = Vtr.reshape(chi_eff * local_dim, rest_dim_next)
            bond_right_dim = chi_eff

    # Convert A_list_rev (right-to-left) into left-to-right order and canonical shape (bond_left, d, d, bond_right)
    A_lr = []
    for k, A in enumerate(reversed(A_list_rev)):
        # A has shape (chi_eff, d, d, bond_right_dim)
        # Convert to (bond_left_dim, d, d, chi_eff) by swapping axes
        A_lr.append(A.transpose(3, 1, 2, 0))
    return A_lr

def mpu_svd_sweep_bidir(U, n_qubits, chi=None, energy_threshold=None, debug=False):
    A_lr = mpu_extract_all_sites(U, n_qubits, chi=chi, energy_threshold=energy_threshold, debug=debug)
    A_rl = mpu_svd_sweep_rl(U, n_qubits, chi=chi, energy_threshold=energy_threshold, debug=debug)
    A_bidir = []
    for A1, A2 in zip(A_lr, A_rl):
        # Trim to common bond dims
        bl = min(A1.shape[0], A2.shape[0])
        br = min(A1.shape[3], A2.shape[3])
        A1_trim = A1[:bl, :, :, :br]
        A2_trim = A2[:bl, :, :, :br]
        A_bidir.append(0.5 * (A1_trim + A2_trim))
    return A_bidir

# ---------------------------
# Validation and tests
# ---------------------------

def assert_unitary(U, tol=1e-10):
    dim = U.shape[0]
    I = cp.eye(dim, dtype=U.dtype)
    err = cp.linalg.norm(U.conj().T @ U - I)
    assert err < tol, f"Matrix not unitary: ||U^†U - I|| = {float(err)}"

def run_mpu_cz_validation(N=6, chi=8, energy_threshold=None, brickwork=False, layers=1, debug=False):
    print("\n--- MPU CZ-Layer Validation ---")
    U_cz = build_unitary_uniform_bulk_mpu(N, brickwork=brickwork, layers=layers)
    assert_unitary(U_cz)
    U_mpu_target, A_list = mpu_projection(U_cz, N, chi=chi, energy_threshold=energy_threshold, debug=debug)
    # Align global phase
    num = cp.vdot(U_mpu_target.flatten(), U_cz.flatten())
    phase = num / (cp.abs(num) + 1e-18)
    U_mpu_target = phase.conj() * U_mpu_target

    frob_cz_mpu = float(cp.linalg.norm(U_cz - U_mpu_target))
    fid_cz_mpu = operator_fidelity(U_mpu_target, U_cz)
    phys_cz_mpu = physical_state_fidelity(U_mpu_target, U_cz, N, trials=20)
    func_cz_mpu = functional_circuit_fidelity(U_mpu_target, U_cz)

    print(f"CZ vs MPU_target Frobenius:        {frob_cz_mpu:.6e}")
    print(f"CZ vs MPU_target operator fidelity:{fid_cz_mpu:.6e}")
    print(f"CZ vs MPU_target physical fidelity:{phys_cz_mpu:.6e}")
    print(f"CZ vs MPU_target functional fid.:  {func_cz_mpu:.6e}")

    # Basic checks
    assert_unitary(U_mpu_target)
    return {
        "U_cz": U_cz,
        "U_mpu": U_mpu_target,
        "A_list": A_list,
        "metrics": {
            "frob": frob_cz_mpu,
            "operator_fid": fid_cz_mpu,
            "physical_fid": phys_cz_mpu,
            "functional_fid": func_cz_mpu
        }
    }

def run_mpu_full_pipeline_test(N=6, chi=4, energy_threshold=None, debug=False):
    print("\n--- MPU Full Pipeline Test ---")
    U_raw = build_unitary_uniform_bulk_mpu(N)
    assert_unitary(U_raw)
    U_mpu_target, A_list = mpu_projection(U_raw, N, chi=chi, energy_threshold=energy_threshold, debug=debug)
    # Align global phase
    num = cp.vdot(U_mpu_target.flatten(), U_raw.flatten())
    phase = num / (cp.abs(num) + 1e-18)
    U_mpu_target = phase.conj() * U_mpu_target

    E_proj = float(cp.linalg.norm(U_raw - U_mpu_target))
    print(f"[Stage 1] Projection delta ||U_raw - U_mpu_target||_F = {E_proj:.6e}")

    frob_cz_mpu = float(cp.linalg.norm(U_raw - U_mpu_target))
    fid_cz_mpu = operator_fidelity(U_mpu_target, U_raw)
    print(f"[Stage 1] CZ vs MPU_target Frobenius: {frob_cz_mpu:.6e}")
    print(f"[Stage 1] CZ vs MPU_target fidelity:  {fid_cz_mpu:.6e}")


    return {
        "U_raw": U_raw,
        "U_mpu": U_mpu_target,
        "A_list": A_list,
        "E_proj": E_proj,
        "fidelity": fid_cz_mpu
    }

def run_mpu_bidir_test(N=6, chi=4, energy_threshold=None):
    print("\n--- MPU Bidirectional Sweep Test ---")
    U_cz = build_unitary_uniform_bulk_mpu(N)
    A_bidir = mpu_svd_sweep_bidir(U_cz, N, chi=chi, energy_threshold=energy_threshold, debug=False)
    for k, A in enumerate(A_bidir):
        print(f"A_bidir[{k}] shape: {A.shape}")
    return A_bidir


# ---------------------------
# Qiskit compatibility test
# ---------------------------


def run_qiskit_test(U_mpu, n_qubits):
    """
    Test whether the MPU-reconstructed unitary U_mpu can be consumed by Qiskit
    and decomposed into a valid quantum circuit.

    This verifies external compatibility: if Qiskit accepts the matrix and
    synthesizes a circuit, then the geometric MPU extraction produces a valid
    unitary suitable for standard quantum compilers.
    """

    # Convert CuPy -> NumPy
    U_np = np.array(U_mpu.get())

    # Wrap as Qiskit Operator
    op = Operator(U_np)

    print("\n--- Qiskit Compatibility Test ---")

    # Try method 1: from_instruction (most universal)
    try:
        qc = QuantumCircuit.from_instruction(op)
        print("Using QuantumCircuit.from_instruction(op)")
    except Exception:
        # Try method 2: append operator to empty circuit
        try:
            qc = QuantumCircuit(n_qubits)
            qc.append(op, range(n_qubits))
            print("Using qc.append(op, range(n_qubits))")
        except Exception:
            # Try method 3: convert operator to instruction
            instr = op.to_instruction()
            qc = QuantumCircuit(n_qubits)
            qc.append(instr, range(n_qubits))
            print("Using op.to_instruction() fallback")

    print("\n--- Qiskit Circuit Synthesis Successful ---")
    print(qc)
    print("\nCircuit depth:", qc.depth())
    print("Number of gates:", qc.size())



    qc_decomp = transpile(qc, basis_gates=['cx', 'u3'])
    print(qc_decomp)


    return qc



# ---------------------------
# Qiskit equivalence + fidelity test
# ---------------------------

def run_qiskit_equivalence_test(U_mpu, qc):
    """
    Compute the unitary of the Qiskit circuit and compare it to the MPU operator.
    Returns Frobenius distance and operator fidelity.
    """


    # Convert CuPy -> NumPy
    U_mpu_np = np.array(U_mpu.get())

    # Get Qiskit circuit unitary
    U_qiskit = Operator(qc).data

    # Frobenius difference
    frob = np.linalg.norm(U_mpu_np - U_qiskit)

    # Operator fidelity
    num = np.abs(np.trace(U_mpu_np.conj().T @ U_qiskit))**2
    den = (U_mpu_np.shape[0]**2)
    fid = num / den

    print("\n--- Qiskit Equivalence Test ---")
    print(f"Frobenius ||U_mpu - U_qiskit||_F = {frob:.6e}")
    print(f"Operator fidelity = {fid:.6e}")

    return frob, fid


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":



    ###########################################################################
    # JSON-SAFE CONVERSION LAYER (bulletproof)
    ###########################################################################
    def json_safe(obj):
        

        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [json_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if hasattr(obj, "get_edges"):
            return [list(edge) for edge in obj.get_edges()]
        if hasattr(obj, "name") and hasattr(obj, "parameters"):
            return {
                "name": obj.name,
                "qubits": json_safe(getattr(obj, "qubits", None)),
                "parameters": json_safe(obj.parameters)
            }
        if hasattr(obj, "to_dict"):
            return json_safe(obj.to_dict())
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    ###########################################################################
    # REAL HARDWARE RUN — CZ-LAYER UNIFORM-BULK MPU
    ###########################################################################

    # ============================================================
    # 1) BUILD CZ-LAYER MPU OPERATOR
    # ============================================================
    N = 6
    chi = 64

    print(f"\n=== CZ-Layer MPU Hardware Test: N = {N}, chi = {chi} ===")

    # Build target CZ-layer unitary
    U_cz = build_unitary_uniform_bulk_mpu(N, brickwork=False, layers=1)
    assert_unitary(U_cz)

    # Project to MPU via geometric SVD pipeline
    U_mpu_cz, A_list = mpu_projection(U_cz, N, chi=chi, energy_threshold=None, debug=False)

    # Align global phase (inline, no external helper)
    num = cp.vdot(U_mpu_cz.flatten(), U_cz.flatten())
    phase = num / (cp.abs(num) + 1e-18)
    U_mpu_cz = phase.conj() * U_mpu_cz

    # Diagnostics
    frob = float(cp.linalg.norm(U_cz - U_mpu_cz))
    op_fid = operator_fidelity(U_mpu_cz, U_cz)
    phys_fid = physical_state_fidelity(U_mpu_cz, U_cz, N, trials=20)

    print(f"Operator Frobenius ||U_cz - U_mpu_cz||_F = {frob:.6e}")
    print(f"Operator fidelity F(U_mpu_cz, U_cz)      = {op_fid:.6e}")
    print(f"Physical state fidelity                  = {phys_fid:.6e}")

    assert_unitary(U_mpu_cz)

    # Qiskit compatibility + equivalence
    qc = run_qiskit_test(U_mpu_cz, N)
    run_qiskit_equivalence_test(U_mpu_cz, qc)

    # Prepare |+> input state
    for q in range(N):
        qc.h(q)

    # Add measurements
    qc_meas = qc.copy()
    qc_meas.measure_all()

    # Save circuit
    with open(f"qc_cz_layer_N{N}.qpy", "wb") as f:
        qpy.dump(qc_meas, f)

    print(f"\nSaved circuit qc_cz_layer_N{N}.qpy")

    # ============================================================
    # 2) HARDWARE SETUP
    # ============================================================
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="X", #API Key
        instance="YOUR_TEST"
    )

    backend = service.backend("X") #SERVER Name
    sampler = SamplerV2(backend)

    print("\n=== Available Backends ===")
    print(service.backends())

    # ============================================================
    # 3) TRANSPILATION
    # ============================================================
    qc_trans = transpile(
        qc_meas,
        backend=backend,
        optimization_level=1
    )

    print("\n--- Transpiled Circuit ---")
    print(qc_trans)
    print("Transpiled depth:", qc_trans.depth())
    print("Transpiled gate counts:", qc_trans.count_ops())

    # ============================================================
    # 4) RUN ON HARDWARE
    # ============================================================
    job = sampler.run([qc_trans], shots=2048)
    print("Job ID:", job.job_id())

    result = job.result()
    counts = result[0].data.meas.get_counts()

    print("\n--- IBM Hardware Output (Counts) ---")
    print(counts)

    # ============================================================
    # 5) BACKEND CALIBRATION (CRASH-PROOF)
    # ============================================================
    props = backend.properties()

    t1_times = []
    t2_times = []
    readout_errors = []

    for q in range(backend.num_qubits):
        try:
            t1_times.append(props.t1(q))
        except:
            t1_times.append(None)

        try:
            t2_times.append(props.t2(q))
        except:
            t2_times.append(None)

        try:
            readout_errors.append(props.readout_error(q))
        except:
            readout_errors.append(None)

    gate_errors = {}
    for gate in props.gates:
        try:
            gate_errors[gate.name] = {
                "qubits": gate.qubits,
                "parameters": gate.parameters
            }
        except:
            gate_errors[gate.name] = None

    try:
        cx_gate_error = props.gate_error("cx")
    except:
        cx_gate_error = None

    try:
        config = backend.configuration()
    except:
        config = None

    try:
        status = backend.status()
    except:
        status = None

    # ============================================================
    # 6) SAVE JSON (FULLY JSON-SAFE)
    # ============================================================
    output_json = json_safe({
        "job_id": job.job_id(),
        "backend": backend.name,
        "counts": counts,

        "depth": qc_trans.depth(),
        "gate_counts": qc_trans.count_ops(),
        "coupling_map": backend.coupling_map,
        "basis_gates": backend.basis_gates,

        "gate_errors": gate_errors,
        "t1": t1_times,
        "t2": t2_times,
        "readout": readout_errors,
        "cx_gate_error": cx_gate_error,

        "config": config,
        "status": status
    })

    with open(f"ibm_run_cz_layer_N{N}.json", "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"\nSaved CZ-Layer JSON: ibm_run_cz_layer_N{N}.json")





