
"""
David Mulnix copyright 2026

This module contains a small suite of reproducibility and validation tests
for the geometric MPU pipeline.  The following tests
are provided and are intended to be run in the order shown in the paper's
methods or in a short reproducibility script.  Each test is annotated with
the precise part of the geometric math it exercises so reviewers can map
printed outputs back to the underlying equations.

Tests included
--------------
1) run_mpu_full_pipeline_test()
   - Purpose: short sanity check on the uniform CZ layer using the full
     extraction → contraction → polar projection pipeline.
   - What it exercises:
       * Global unitary → interleaved tensor representation:
         U ∈ U(2^N) reshaped to U_{i1 j1 i2 j2 ... iN jN}.
       * Site-wise SVD extraction on the interleaved matrix M^{(1)}.
       * Bond-dimension propagation and truncated extraction (chi).
       * Tensor contraction reconstruction and polar re-unitarization.
       * Operator fidelity and Frobenius distance checks.
   - Expected outputs: small projection delta ||U_raw - U_mpu||_F,
     operator fidelity ≈ 1.0.  Use as a canonical pipeline sanity check.


2) run_mpu_bidir_test()
   - Purpose: demonstrate bidirectional gauge stabilization and produce
     averaged tensors for gauge robustness.
   - What it exercises:
       * Right→left SVD sweep (mirror of left→right) and conversion to
         canonical left-to-right ordering.
       * Gauge averaging: (A_lr + A_rl)/2 to reduce gauge drift.
   - Expected outputs: A_bidir shapes and confirmation that averaging
     preserves the expected bond profile.  Use this to justify gauge
     stabilization in the methods.

3) run_nonuniform_mpu_test()
   - Purpose: the primary result for the paper — validate the pipeline on
     a non-uniform nearest-neighbor two-qubit layer (different gates on
     different bonds).
   - What it exercises:
       * All geometric steps end-to-end on a non-uniform target:
         - interleaving reshape,
         - site-wise SVD extraction (adaptive or fixed chi),
         - bond propagation,
         - contraction to T,
         - polar re-unitarization to nearest unitary,
         - global phase alignment for comparisons.
       * Validation metrics: Frobenius norm, operator fidelity,
         randomized action-on-states tests, and reconstruction fidelity.
   - Expected outputs: Frobenius ≈ machine precision, operator fidelity = 1.0,
     action-on-states differences ≲ 1e-14.  This test is the headline result:
     it demonstrates that the geometric MPU method recovers non-uniform MPU
     operators in code.

Mathematical mapping 
----------------------------------------------------
- Interleaved tensor representation:
    U_{i1 j1 i2 j2 ... iN jN}  ← reshape(U) and transpose to interleave
  Explain why interleaving places the global unitary in a coordinate chart
  suitable for local SVD extraction.

- Site-wise SVD extraction:
    M^{(k)} = U^{(k)} Σ^{(k)} V^{(k)†}
    A^{(k)}_{α,i,j,β} = U^{(k)}_{α i j, β} · Σ^{(k)}_{β}
  Show the indexing convention and how rows map to (bond_left × i × j).

- Bond-dimension propagation:
    bond_left^{(k+1)} = χ_eff^{(k)}
  Explain truncation policy (fixed χ or energy threshold) and report
  singular-value spectra and energy_kept per site in supplementary material.

- Tensor contraction reconstruction:
    T = A^{(1)} ⋆ A^{(2)} ⋆ ... ⋆ A^{(N)}
    reorder axes → reshape → U_mpu (matrix)
  State the exact contraction ordering and the permutation used to obtain
  (i1...iN, j1...jN) before reshaping to a 2^N × 2^N matrix.

- Polar re-unitarization:
    U_mpu_unitary = nearest_unitary(T) via SVD: T = U S V† → U V†
  Justify as nearest unitary in Frobenius norm and show the SVD formula.

- Global phase alignment:
    φ = ⟨U_mpu, U_target⟩ / |⟨U_mpu, U_target⟩|
    U_mpu ← φ* · U_mpu
  Explain that global phase is physically irrelevant and required for
  meaningful fidelity comparisons.

- Geometric fidelity metrics:
    Operator fidelity: |Tr(U_mpu† U_target)|^2 / (2^N)^2
    State fidelity: |⟨U_mpu ψ | U_target ψ⟩|^2
    Frobenius distance: ||U_mpu - U_target||_F
  State which metrics are primary (operator fidelity, Frobenius) and which
  are supporting (randomized state-action tests).

"""


import cupy as cp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from module_fidelity_objective import FidelityObjective
import json
from qiskit import qpy, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# ============================================================
# Utility Metrics (from your structured tests module)
# ============================================================

def frobenius_diff(A, B):
    return float(cp.linalg.norm(A - B))

def delta_norm(A, B):
    return float(cp.linalg.norm(A - B))


def functional_circuit_fidelity(Ua, Ub, n_qubits):
    # Functional fidelity via measurement probabilities
    # (purely geometric / spectral)
    probs_a = cp.abs(Ua[:, 0])**2
    probs_b = cp.abs(Ub[:, 0])**2
    return float(cp.sum(cp.sqrt(probs_a * probs_b)))


# ============================================================
# CZ-layer builder for MPU
# ============================================================

def build_unitary_uniform_bulk_mpu(N):
    I2 = cp.eye(2, dtype=cp.complex128)

    CZ = cp.asarray([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=cp.complex128)

    dim = 2**N
    U = cp.eye(dim, dtype=cp.complex128)

    def embed_two_qubit_gate(G, q, N):
        U_full = cp.eye(1, dtype=cp.complex128)
        k = 0
        while k < N:
            if k == q:
                U_full = cp.kron(U_full, G)
                k += 2
            else:
                U_full = cp.kron(U_full, I2)
                k += 1
        return U_full

    for layer in range(N - 1):
        for q in range(N - 1):
            G = embed_two_qubit_gate(CZ, q, N)
            U = G @ U

    return U


def build_nonuniform_mpu_layer(N=6):
    I2 = cp.eye(2, dtype=cp.complex128)

    # Define 2-qubit gates using pure Python scalars
    CZ = cp.array([[1+0j, 0, 0, 0],
                   [0, 1+0j, 0, 0],
                   [0, 0, 1+0j, 0],
                   [0, 0, 0, -1+0j]], dtype=cp.complex128)

    CP_pi3 = cp.array([[1+0j, 0, 0, 0],
                       [0, 1+0j, 0, 0],
                       [0, 0, 1+0j, 0],
                       [0, 0, 0, complex(cp.exp(1j * (np.pi/3)))]],
                      dtype=cp.complex128)

    CY = cp.array([[1+0j, 0, 0, 0],
                   [0, 1+0j, 0, 0],
                   [0, 0, 0, -1j],
                   [0, 0, 1j, 0]], dtype=cp.complex128)

    CX = cp.array([[1+0j, 0, 0, 0],
                   [0, 1+0j, 0, 0],
                   [0, 0, 0, 1+0j],
                   [0, 0, 1+0j, 0]], dtype=cp.complex128)

    CZ2 = CZ

    gates = [CZ, CP_pi3, CY, CX, CZ2]

    def embed_two_qubit_gate(G, q, N):
        U_full = cp.eye(1, dtype=cp.complex128)
        k = 0
        while k < N:
            if k == q:
                U_full = cp.kron(U_full, G)
                k += 2
            else:
                U_full = cp.kron(U_full, I2)
                k += 1
        return U_full

    dim = 2**N
    U = cp.eye(dim, dtype=cp.complex128)

    for q, G in enumerate(gates):
        U = embed_two_qubit_gate(G, q, N) @ U

    return U



# ============================================================
# MPU extraction sweep
# ============================================================

def mpu_extract_all_sites(U, n_qubits, chi, debug=False):
    d = 2
    N = n_qubits

    U_tensor = U.reshape((d,)*N + (d,)*N)
    interleave = []
    for k in range(N):
        interleave.append(k)
        interleave.append(N+k)
    U_inter = U_tensor.transpose(interleave)

    local_dim = d**2
    rest_dim = d**(2*(N-1))
    M = U_inter.reshape(local_dim, rest_dim)

    A_list = []
    bond_left_dim = 1

    for site in range(N):
        Umat, Svec, Vh = cp.linalg.svd(M, full_matrices=False)
        chi_eff = min(chi, Umat.shape[1])

        Utr = Umat[:, :chi_eff]
        Str = Svec[:chi_eff]
        Vtr = Vh[:chi_eff, :]

        A_site = cp.zeros((bond_left_dim, d, d, chi_eff), dtype=cp.complex128)
        for alpha in range(bond_left_dim):
            for i in range(d):
                for j in range(d):
                    idx = alpha*d**2 + i*d + j
                    for beta in range(chi_eff):
                        A_site[alpha, i, j, beta] = Utr[idx, beta] * Str[beta]

        A_list.append(A_site)

        if site < N-1:
            rest_dim_next = d**(2*(N-site-2))
            M = Vtr.reshape(chi_eff * d**2, rest_dim_next)
            bond_left_dim = chi_eff

    if debug:
        print("\n--- MPU Multi-Site Extraction ---")
        for k, A in enumerate(A_list):
            print(f"A[{k}] shape: {A.shape}")

    return A_list


# ============================================================
# MPU projection (SVD-based)
# ============================================================

def mpu_projection(U, n_qubits, chi, debug=False):
    N = n_qubits
    dim = 2**N

    A_list = mpu_extract_all_sites(U, N, chi, debug=debug)

    T = A_list[0]
    for k in range(1, N):
        Ak = A_list[k]

        m = min(T.shape[-1], Ak.shape[0])
        T = T[..., :m]
        Ak = Ak[:m, :, :, :]

        T = cp.tensordot(T, Ak, axes=([-1], [0]))

    l = cp.ones((T.shape[0],), dtype=cp.complex128)
    r = cp.ones((T.shape[-1],), dtype=cp.complex128)

    T = cp.tensordot(l.conj(), T, axes=([0], [0]))
    T = cp.tensordot(T, r, axes=([-1], [0]))

    axes = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U_mpu = T.transpose(axes).reshape(dim, dim)

    Ux, Sx, Vxh = cp.linalg.svd(U_mpu, full_matrices=False)
    return Ux @ Vxh


def align_global_phase(U_ref, U):
    num = cp.vdot(U.flatten(), U_ref.flatten())
    phase = num / cp.abs(num + 1e-18)
    return phase * U


# ============================================================
# MPU energy functional
# ============================================================

def mpu_energy(U, U_mpu):
    return float(cp.linalg.norm(U - U_mpu))


# ============================================================
# MPU validation and pipeline tests
# ============================================================

def run_mpu_full_pipeline_test():
    print("\n--- MPU Full Pipeline Test ---")

    N = 6
    U_raw = build_unitary_uniform_bulk_mpu(N)

    chi = 4
    U_mpu_target = mpu_projection(U_raw, N, chi, debug=False)
    U_mpu_target = align_global_phase(U_raw, U_mpu_target)

    E_proj = float(cp.linalg.norm(U_raw - U_mpu_target))
    print(f"[Stage 1] Projection delta ||U_raw - U_mpu_target||_F = {E_proj:.6f}")

    frob_cz_mpu = float(cp.linalg.norm(U_raw - U_mpu_target))
    fid_cz_mpu = FidelityObjective.operator_fidelity(U_mpu_target, U_raw)
    print(f"[Stage 1] CZ vs MPU_target Frobenius: {frob_cz_mpu:.6f}")
    print(f"[Stage 1] CZ vs MPU_target fidelity:  {fid_cz_mpu:.6f}")

# ============================================================
# BIDIRECTIONAL MPU SVD SWEEP
# ============================================================

def mpu_svd_sweep_lr(U, n_qubits, chi):
    d = 2
    N = n_qubits

    U_tensor = U.reshape((d,)*N + (d,)*N)
    interleave = []
    for k in range(N):
        interleave.append(k)
        interleave.append(N+k)
    U_inter = U_tensor.transpose(interleave)

    local_dim = d**2
    rest_dim = d**(2*(N-1))
    M = U_inter.reshape(local_dim, rest_dim)

    A_list = []
    bond_left_dim = 1

    for site in range(N):
        Umat, Svec, Vh = cp.linalg.svd(M, full_matrices=False)
        chi_eff = min(chi, Umat.shape[1])

        Utr = Umat[:, :chi_eff]
        Str = Svec[:chi_eff]
        Vtr = Vh[:chi_eff, :]

        A_site = cp.zeros((bond_left_dim, d, d, chi_eff), dtype=cp.complex128)
        for alpha in range(bond_left_dim):
            for i in range(d):
                for j in range(d):
                    idx = alpha*d**2 + i*d + j
                    for beta in range(chi_eff):
                        A_site[alpha, i, j, beta] = Utr[idx, beta] * Str[beta]

        A_list.append(A_site)

        if site < N-1:
            rest_dim_next = d**(2*(N-site-2))
            M = Vtr.reshape(chi_eff * d**2, rest_dim_next)
            bond_left_dim = chi_eff

    return A_list


def mpu_svd_sweep_rl(U, n_qubits, chi):
    d = 2
    N = n_qubits

    U_tensor = U.reshape((d,)*N + (d,)*N)
    interleave = []
    for k in range(N):
        interleave.append(k)
        interleave.append(N+k)
    U_inter = U_tensor.transpose(interleave)

    axes = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U_rev = U_inter.transpose(axes[::-1])

    local_dim = d**2
    rest_dim = d**(2*(N-1))
    M = U_rev.reshape(local_dim, rest_dim)

    A_list_rev = []
    bond_right_dim = 1

    for site in range(N):
        Umat, Svec, Vh = cp.linalg.svd(M, full_matrices=False)
        chi_eff = min(chi, Umat.shape[1])

        Utr = Umat[:, :chi_eff]
        Str = Svec[:chi_eff]
        Vtr = Vh[:chi_eff, :]

        A_site = cp.zeros((chi_eff, d, d, bond_right_dim), dtype=cp.complex128)
        for beta in range(bond_right_dim):
            for i in range(d):
                for j in range(d):
                    idx = i*d + j
                    for alpha in range(chi_eff):
                        A_site[alpha, i, j, beta] = Utr[idx, alpha] * Str[alpha]

        A_list_rev.append(A_site)

        if site < N-1:
            rest_dim_next = d**(2*(N-site-2))
            M = Vtr.reshape(chi_eff * d**2, rest_dim_next)
            bond_right_dim = chi_eff

    return A_list_rev[::-1]


def mpu_svd_sweep_bidir(U, n_qubits, chi):
    A_lr = mpu_svd_sweep_lr(U, n_qubits, chi)
    A_rl = mpu_svd_sweep_rl(U, n_qubits, chi)

    A_bidir = []
    for A1, A2 in zip(A_lr, A_rl):
        bl = min(A1.shape[0], A2.shape[0])
        br = min(A1.shape[3], A2.shape[3])
        A1_trim = A1[:bl, :, :, :br]
        A2_trim = A2[:bl, :, :, :br]
        A_bidir.append(0.5 * (A1_trim + A2_trim))

    return A_bidir


def run_mpu_bidir_test():
    print("\n--- MPU Bidirectional Sweep Test ---")
    N = 6
    chi = 4
    U_cz = build_unitary_uniform_bulk_mpu(N)

    A_bidir = mpu_svd_sweep_bidir(U_cz, N, chi)

    for k, A in enumerate(A_bidir):
        print(f"A_bidir[{k}] shape: {A.shape}")

def validate_action_on_states(U_target, U_mpu, n_qubits, trials=20):
    print("\n--- Action-on-States Validation ---")
    dim = 2**n_qubits

    for t in range(trials):
        psi = cp.random.randn(dim) + 1j * cp.random.randn(dim)
        psi /= cp.linalg.norm(psi)

        out_target = U_target @ psi
        out_mpu = U_mpu @ psi

        diff = cp.linalg.norm(out_target - out_mpu)
        print(f"Trial {t}: ||U_target ψ - U_mpu ψ|| = {float(diff):.6e}")

def validate_schmidt_ranks(U, n_qubits):
    print("\n--- Schmidt Rank / Bond Dimension Validation ---")

    for cut in range(1, n_qubits):
        left_dim = 2**cut
        right_dim = 2**(n_qubits - cut)

        # Reshape into bipartition
        U_reshaped = U.reshape(left_dim, right_dim, left_dim, right_dim)

        # Permute to (left_in, left_out) × (right_in, right_out)
        U_perm = cp.transpose(U_reshaped, (0, 2, 1, 3))

        # Flatten into matrix for Schmidt rank
        M = U_perm.reshape(left_dim**2, right_dim**2)

        rank = cp.linalg.matrix_rank(M)
        print(f"Cut {cut}: Schmidt rank = {int(rank)}")


def reconstruct_from_A(A_list):
    """
    Reconstruct full unitary from A-tensors.
    A_list[k] shape: (bond_left, 2, 2, bond_right)
    """
    N = len(A_list)

    # Start with first tensor
    T = A_list[0]

    # Contract chain left → right
    for k in range(1, N):
        Ak = A_list[k]

        # Match bond dimensions
        m = min(T.shape[-1], Ak.shape[0])
        T = T[..., :m]
        Ak = Ak[:m, :, :, :]

        # Contract right bond of T with left bond of Ak
        T = cp.tensordot(T, Ak, axes=([-1], [0]))

    # Contract left boundary (bond dimension 1)
    l = cp.ones((T.shape[0],), dtype=cp.complex128)
    T = cp.tensordot(l.conj(), T, axes=([0], [0]))

    # Contract right boundary (bond dimension 1)
    r = cp.ones((T.shape[-1],), dtype=cp.complex128)
    T = cp.tensordot(T, r, axes=([-1], [0]))

    # Reorder physical indices: (in0, out0, in1, out1, ..., inN-1, outN-1)
    axes = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U = T.transpose(axes).reshape(2**N, 2**N)

    return U



def validate_reconstruction(U_target, A_list):
    print("\n--- Reconstruction Consistency Validation ---")
    U_rec = reconstruct_from_A(A_list)

    diff = cp.linalg.norm(U_target - U_rec)
    fid = FidelityObjective.operator_fidelity(U_rec, U_target)

    print(f"Reconstruction Frobenius: {float(diff):.6e}")
    print(f"Reconstruction Fidelity:  {float(fid):.6f}")



def run_nonuniform_mpu_test():
    print("\n--- Non-Uniform MPU Test ---")

    N = 6
    U_non = build_nonuniform_mpu_layer(N)

    chi = 8
    U_mpu = mpu_projection(U_non, N, chi)
    U_mpu = align_global_phase(U_non, U_mpu)

    print(f"Non-uniform Frobenius: {float(cp.linalg.norm(U_non - U_mpu)):.6f}")
    print(f"Non-uniform operator fidelity: {FidelityObjective.operator_fidelity(U_mpu, U_non):.6f}")

    A_list = mpu_extract_all_sites(U_non, N, chi, debug=True)

    validate_action_on_states(U_non, U_mpu, N)
    validate_schmidt_ranks(U_mpu, N)
    validate_reconstruction(U_non, A_list)

# ============================================================
# Qiskit Compatibility Test (Circuit Synthesis)
# ============================================================

def run_qiskit_test(U_mpu, n_qubits):


    print("\n--- Qiskit Compatibility Test ---")

    U_np = np.array(U_mpu.get())
    op = Operator(U_np)

    try:
        instr = op.to_instruction()
        qc = QuantumCircuit(n_qubits)
        qc.append(instr, range(n_qubits))
        print("Using op.to_instruction()")
    except Exception:
        raise ValueError("Qiskit could not synthesize circuit: operator not unitary.")

    print("\n--- Qiskit Circuit Synthesis Successful ---")
    print(qc)
    print("\nCircuit depth:", qc.depth())
    print("Number of gates:", qc.size())

    return qc



# ============================================================
# Qiskit Equivalence Test (U_mpu vs U_qiskit)
# ============================================================

def run_qiskit_equivalence_test(U_mpu, qc):


    print("\n--- Qiskit Equivalence Test ---")

    U_mpu_np = np.array(U_mpu.get())
    U_qiskit = Operator(qc).data

    frob = np.linalg.norm(U_mpu_np - U_qiskit)
    fid = (np.abs(np.trace(U_mpu_np.conj().T @ U_qiskit))**2) / (U_mpu_np.shape[0]**2)

    print(f"Frobenius ||U_mpu - U_qiskit||_F = {frob:.6e}")
    print(f"Operator fidelity = {fid:.6e}")

    return frob, fid


def polar_unitary(M):
    U, S, Vh = cp.linalg.svd(M, full_matrices=False)
    return U @ Vh


def run_qiskit_decomposition(U_mpu, n_qubits):
    """
    Decompose U_mpu into a CX+U3 gate sequence via transpilation.
    This shows that the MPU operator can be fully expressed in a
    standard hardware-native gate set.
    """

    print("\n--- Qiskit Decomposition Test (CX + U3) ---")

    U_np = np.array(U_mpu.get())
    op = Operator(U_np)

    instr = op.to_instruction()
    qc = QuantumCircuit(n_qubits)
    qc.append(instr, range(n_qubits))

    # Transpile into CX+U3 basis
    qc_decomp = transpile(qc, basis_gates=["cx", "u3"], optimization_level=1)

    print(qc_decomp)
    print("\nDecomposed circuit depth:", qc_decomp.depth())
    print("Decomposed circuit gate count:", qc_decomp.size())

    return qc_decomp

def run_qiskit_decomposition_equivalence(U_mpu, qc_decomp):
    print("\n--- Qiskit Decomposition Equivalence Test ---")

    U_mpu_np = np.array(U_mpu.get())
    U_qiskit = Operator(qc_decomp).data

    frob = np.linalg.norm(U_mpu_np - U_qiskit)
    fid = (np.abs(np.trace(U_mpu_np.conj().T @ U_qiskit))**2) / (U_mpu_np.shape[0]**2)

    print(f"Frobenius ||U_mpu - U_qiskit||_F = {frob:.6e}")
    print(f"Operator fidelity = {fid:.6e}")

    return frob, fid

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
    # REAL HARDWARE RUN — NON-UNIFORM MPU
    ###########################################################################

    # ============================================================
    # 1) BUILD NON-UNIFORM MPU OPERATOR
    # ============================================================
    N = 6
    chi = 64

    print(f"\n=== Non-Uniform MPU Test: N = {N} ===")

    U_non = build_nonuniform_mpu_layer(N)
    U_mpu_non = mpu_projection(U_non, N, chi)
    U_mpu_non = align_global_phase(U_non, U_mpu_non)

    print(f"Operator Frobenius ||U_non - U_mpu||_F = {float(cp.linalg.norm(U_non - U_mpu_non)):.6e}")
    print(f"Operator fidelity F(U_mpu, U_non) = {FidelityObjective.operator_fidelity(U_mpu_non, U_non):.6f}")

    # Qiskit compatibility
    qc = run_qiskit_test(U_mpu_non, N)
    run_qiskit_equivalence_test(U_mpu_non, qc)

    # CX+U3 decomposition
    qc_decomp = run_qiskit_decomposition(U_mpu_non, N)
    run_qiskit_decomposition_equivalence(U_mpu_non, qc_decomp)

    # Prepare |+> input state
    for q in range(N):
        qc_decomp.h(q)

    # Add measurements
    qc_meas = qc_decomp.copy()
    qc_meas.measure_all()

    # Save circuit
    with open("qc_nonuniform_N6.qpy", "wb") as f:
        qpy.dump(qc_meas, f)

    print("\nSaved circuit qc_nonuniform_N6.qpy")

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
        try: t1_times.append(props.t1(q))
        except: t1_times.append(None)

        try: t2_times.append(props.t2(q))
        except: t2_times.append(None)

        try: readout_errors.append(props.readout_error(q))
        except: readout_errors.append(None)

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

    with open("ibm_run_nonuniform_N6.json", "w") as f:
        json.dump(output_json, f, indent=2)

    print("\nSaved Non-Uniform JSON: ibm_run_nonuniform_N6.json")
