# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026

MPU QFT Test: Purpose, Geometric Math, and How to Read the Results

This script implements the MPU–QFT use case using a purely geometric
extraction–projection pipeline. In the MPU paper, the Quantum Fourier Transform
(QFT) is one of the canonical “high‑bond” and “deep‑structure” examples: its
entanglement profile grows exponentially across the chain, and its MPU
representation requires large internal bond dimensions. Algebraically, the
authors describe this in terms of nonuniform bulk tensors and deep
isometry‑merging trees. Here, instead of using their algebraic construction, we
start from the full global QFT unitary and recover its MPU structure directly
through geometric tensor methods.

Geometric viewpoint
-------------------
The geometric MPU method treats the global QFT operator U ∈ U(2^N) as a
rank‑2N tensor, interleaves its input/output indices, and performs a site‑wise
SVD factorization. Each SVD produces a local tensor A^{(k)} whose bond
dimensions reflect the entanglement structure at that site. For QFT, these
bond dimensions grow rapidly (4, 16, 64, 256, 1024, …), matching the known
high‑bond profile of QFT. The geometric method “sees” QFT not as a special
algebraic object but as a global unitary whose singular‑value structure
naturally induces large internal bonds.

The geometric pipeline consists of:
  • Interleaving U into U_{i1 j1 i2 j2 … iN jN}
  • Site‑wise SVD extraction:
        A^{(k)}_{α,i,j,β} = U^{(k)}_{α i j, β} · Σ^{(k)}_{β}
  • Bond‑dimension propagation: bond_left → χ_eff
  • Tensor contraction of A‑tensors along bond indices
  • Polar re‑unitarization (nearest unitary)
  • Global phase alignment
  • Validation via operator fidelity, action‑on‑states, Schmidt ranks,
    basis‑state tests, inverse composition, and shift‑operator diagonalization

This is entirely geometric math. No algebraic MPU rules from the original paper
are used.

What this test does
-------------------
1. Builds the exact QFT matrix U_qft using the standard formula
       U_{m,k} = ω^{km} / √(2^N)
   and uses it as the target operator.

2. Applies the geometric MPU projection with a large χ (e.g., 1024) to extract
   A‑tensors and reconstruct U_mpu.

3. Validates correctness through:
   • Frobenius(U_qft − U_mpu)
   • Operator fidelity(U_mpu, U_qft)
   • A‑tensor shapes (showing exponential bond growth)
   • Action‑on‑states differences (≈ 10⁻¹¹ for N=10)
   • Schmidt ranks across bipartitions (4,16,64,256,1024,…)
   • Reconstruction fidelity
   • Basis‑state tests
   • Inverse composition U_qft^{-1} U_mpu ≈ I
   • Diagonalization of the shift operator S

These checks confirm that the geometric MPU method reconstructs QFT exactly
(up to global phase) and preserves its functional behavior.

How to read the results
-----------------------
• QFT Frobenius = 0.0  
  → The projected MPU operator matches the true QFT matrix up to numerical
    precision.

• Operator fidelity = 1.0  
  → Exact equality up to global phase.

• Action‑on‑states ≈ 10⁻¹¹  
  → For a 1024×1024 unitary, this is excellent numerical agreement.

• Schmidt ranks = {4,16,64,256,1024,…}  
  → This is the expected high‑bond profile of QFT.

• Basis‑state max diff ≈ 10⁻¹¹  
  → QFT basis action is preserved.

• Inverse composition error ≈ 10⁻¹⁰  
  → U_mpu behaves as a true QFT under inversion.

• Shift diagonalization off‑diagonal ≈ 10⁻¹¹  
  → QFT’s defining spectral property is preserved.

Important note about reconstruction Frobenius
---------------------------------------------
When reconstructing U_rec from A‑tensors, Frobenius behaves differently:

• In the QFT test, U_rec and U_target differ by a *global phase*.  
• Frobenius norm is sensitive to global phase, even though fidelity is not.  
• For a 1024×1024 unitary, a pure global phase can produce Frobenius ≈ 64.  
• Fidelity = 1.0 proves the reconstruction is exact (up to phase).  
• Therefore:
      – Frobenius ≈ 64 is NOT an error in this context.
      – Fidelity is the correct metric for reconstruction.
      – In other parts of the test (projection), Frobenius = 0 IS expected.

Summary
-------
This test demonstrates that the geometric MPU extraction–projection pipeline
successfully reconstructs the QFT operator—one of the most demanding high‑bond
MPU examples—with operator fidelity 1.0 and full functional correctness. These
results show that QFT can be represented as an MPU using geometric tensor
methods, without relying on the algebraic MPU machinery.
"""


import cupy as cp
from module_fidelity_objective import FidelityObjective
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit import transpile


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
    
def debug_reconstruction(U_target, U_mpu, A_list):
    print("\n--- Reconstruction Debug ---")

    U_rec = reconstruct_from_A(A_list)

    diff_target = cp.linalg.norm(U_target - U_rec)
    diff_mpu    = cp.linalg.norm(U_mpu    - U_rec)

    fid_target = FidelityObjective.operator_fidelity(U_rec, U_target)
    fid_mpu    = FidelityObjective.operator_fidelity(U_rec, U_mpu)

    print(f"Frob(U_rec - U_target): {float(diff_target):.6e}")
    print(f"Frob(U_rec - U_mpu):    {float(diff_mpu):.6e}")
    print(f"Fid(U_rec, U_target):   {float(fid_target):.6f}")
    print(f"Fid(U_rec, U_mpu):      {float(fid_mpu):.6f}")

def build_mpu_qft_variant(N=6):
    I2 = cp.eye(2, dtype=cp.complex128)

    def H():
        return (1 / cp.sqrt(2)) * cp.array([
            [1+0j, 1+0j],
            [1+0j, -1+0j]
        ], dtype=cp.complex128)

    def CP(theta):
        return cp.array([
            [1+0j, 0, 0, 0],
            [0, 1+0j, 0, 0],
            [0, 0, 1+0j, 0],
            [0, 0, 0, complex(cp.exp(1j * theta))]
        ], dtype=cp.complex128)

    def embed_single_qubit_gate(G, q, N):
        U_full = cp.eye(1, dtype=cp.complex128)
        for k in range(N):
            if k == q:
                U_full = cp.kron(U_full, G)
            else:
                U_full = cp.kron(U_full, I2)
        return U_full

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

    # QFT: H on k, then CP(k->j) for j>k
    for k in range(N):
        U = embed_single_qubit_gate(H(), k, N) @ U

        for j in range(k+1, N):
            theta = cp.pi / (2 ** (j - k))
            G = CP(theta)
            U = embed_two_qubit_gate(G, k, N) @ U

    return U

def build_reference_qft(N):
    dim = 2**N
    omega = cp.exp(2j * cp.pi / dim)
    U = cp.zeros((dim, dim), dtype=cp.complex128)
    for k in range(dim):
        for m in range(dim):
            U[m, k] = omega**(k * m) / cp.sqrt(dim)
    return U


def run_mpu_qft_test():
    print("\n--- MPU QFT Variant Test ---")

    N = 10 #6, 8
    chi = 1024 #64, 256

    # USE TRUE QFT MATRIX AS TARGET
    U_qft = build_reference_qft(N)

    U_mpu = mpu_projection(U_qft, N, chi)
    U_mpu = align_global_phase(U_qft, U_mpu)

    print(f"QFT Frobenius: {float(cp.linalg.norm(U_qft - U_mpu)):.6f}")
    print(f"QFT operator fidelity: {FidelityObjective.operator_fidelity(U_mpu, U_qft):.6f}")

    A_list = mpu_extract_all_sites(U_qft, N, chi, debug=True)

    validate_action_on_states(U_qft, U_mpu, N)
    validate_schmidt_ranks(U_mpu, N)
    validate_reconstruction(U_qft, A_list)

    validate_qft_on_basis(U_mpu, N)
    validate_qft_inverse(U_mpu, N)
    validate_qft_diagonalizes_shift(U_mpu, N)

def validate_qft_on_basis(U_mpu, N):
    U_ref = build_reference_qft(N)

    dim = 2**N
    max_diff = 0.0

    for k in range(dim):
        basis = cp.zeros(dim, dtype=cp.complex128)
        basis[k] = 1.0

        psi_mpu = U_mpu @ basis
        psi_ref = U_ref @ basis

        diff = cp.linalg.norm(psi_mpu - psi_ref)
        max_diff = max(max_diff, float(diff))

    print(f"QFT basis-state max ||U_mpu|k> - U_ref|k>|| = {max_diff:.6e}")

def validate_qft_inverse(U_mpu, N):
    U_ref = build_reference_qft(N)
    U_inv = U_ref.conj().T

    U_comp = U_inv @ U_mpu
    I = cp.eye(2**N, dtype=cp.complex128)

    diff = cp.linalg.norm(U_comp - I)
    print(f"QFT inverse composition ||U_inv U_mpu - I|| = {float(diff):.6e}")

def build_shift_operator(N):
    dim = 2**N
    S = cp.zeros((dim, dim), dtype=cp.complex128)
    for k in range(dim):
        S[(k+1) % dim, k] = 1.0
    return S

def validate_qft_diagonalizes_shift(U_mpu, N):
    S = build_shift_operator(N)
    U = U_mpu

    D = U @ S @ U.conj().T

    off_diag = D - cp.diag(cp.diag(D))
    norm_off = cp.linalg.norm(off_diag)

    print(f"QFT diagonalization ||off-diagonal(U S U†)|| = {float(norm_off):.6e}")


# ============================================================
# Qiskit Compatibility Test (UnitaryGate wrapper)
# ============================================================

def run_qiskit_test(U_mpu, n_qubits):
    print("\n--- Qiskit Compatibility Test ---")

    U_np = np.array(U_mpu.get())
    op = Operator(U_np)

    instr = op.to_instruction()
    qc = QuantumCircuit(n_qubits)
    qc.append(instr, range(n_qubits))

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

# ============================================================
# Qiskit Decomposition Test (CX + U3)
# ============================================================

def run_qiskit_decomposition(U_mpu, n_qubits):
    print("\n--- Qiskit Decomposition Test (CX + U3) ---")

    U_np = np.array(U_mpu.get())
    op = Operator(U_np)

    instr = op.to_instruction()
    qc = QuantumCircuit(n_qubits)
    qc.append(instr, range(n_qubits))

    # Decompose into hardware-native gates
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
    # Run geometric MPU QFT test
    run_mpu_qft_test()

    # Build QFT target and MPU projection
    N = 10
    chi = 1024
    U_qft = build_reference_qft(N)
    U_mpu_qft = mpu_projection(U_qft, N, chi)
    U_mpu_qft = align_global_phase(U_qft, U_mpu_qft)

    # 1) Qiskit compatibility test
    qc = run_qiskit_test(U_mpu_qft, N)
    run_qiskit_equivalence_test(U_mpu_qft, qc)

    # 2) Full CX+U3 decomposition
    qc_decomp = run_qiskit_decomposition(U_mpu_qft, N)
    run_qiskit_decomposition_equivalence(U_mpu_qft, qc_decomp)

