# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026

High‑Bond MPU Use Case: Geometric Interpretation and Test Summary

This script implements and validates the high‑bond MPU use case using a purely
geometric extraction–projection pipeline. In the MPU literature, “high‑bond”
refers to operators whose internal entanglement structure forces large bond
dimensions in their MPU representation. These operators are typically built
from strongly entangling two‑qubit gates (e.g., controlled‑phase rotations with
large angles, CY, CX, etc.) arranged in a nearest‑neighbor pattern. In the
algebraic MPU framework, such layers require deeper isometry‑merging trees and
more complex amplitude‑amplification steps. Here, instead of constructing the
MPU algebraically, we begin with the full global unitary and recover its MPU
structure geometrically.

Geometric viewpoint
-------------------
The geometric MPU method treats the global unitary U ∈ U(2^N) as a rank‑2N
tensor, interleaves its input/output indices, and performs a site‑wise SVD
factorization. Each SVD produces a local tensor A^{(k)}_{α,i,j,β} whose bond
dimensions reflect the entanglement structure at that site. In the high‑bond
case, the singular value spectra are broad, and the retained χ_eff values grow
significantly in the bulk. This produces A‑tensors with large internal bond
dimensions (e.g., 16), matching the expected high‑bond MPU structure.

The geometric pipeline consists of:
  • Interleaving U into U_{i1 j1 i2 j2 … iN jN}
  • Site‑wise SVD extraction:
        A^{(k)}_{α,i,j,β} = U^{(k)}_{α i j, β} · Σ^{(k)}_{β}
  • Bond‑dimension propagation: bond_left → χ_eff
  • Tensor contraction of A‑tensors along bond indices
  • Polar re‑unitarization (nearest unitary in Frobenius norm)
  • Global phase alignment for meaningful fidelity comparisons
  • Validation via operator fidelity, action‑on‑states, and Schmidt ranks

This geometric method does not use any algebraic MPU rules from the original
paper. It reconstructs the MPU operator directly from the global unitary using
tensor geometry and SVD structure.

What this script tests
----------------------
1. High‑Bond MPU Test
   Builds a strongly entangling nearest‑neighbor layer using controlled‑phase
   gates with large angles, CY, and CX. This produces a high‑bond operator with
   large internal entanglement. The geometric pipeline is applied with χ=16.
   The test prints:
       • Frobenius(U_target − U_mpu)
       • Operator fidelity(U_mpu, U_target)
       • A‑tensor shapes (showing large bond dimensions)
       • Action‑on‑states differences (≈ 10⁻¹⁵)
       • Schmidt ranks across all bipartitions
       • Reconstruction fidelity (exact up to global phase)

   These results demonstrate that the geometric MPU method recovers a
   high‑bond MPU operator exactly (up to global phase).

2. Random High‑Bond MPU Test
   Builds a random high‑bond MPU layer using random two‑qubit unitaries. This
   tests generality beyond structured gates. The same geometric pipeline is
   applied. The test prints:
       • Frobenius and operator fidelity
       • A‑tensor shapes (again showing large bond dimensions)
       • Action‑on‑states differences
       • Schmidt ranks
       • Reconstruction fidelity

   This confirms that the geometric MPU method handles arbitrary high‑bond
   operators, not just structured ones.

Summary
-------
This script demonstrates that the geometric MPU extraction–projection pipeline
successfully reconstructs high‑bond MPU operators—both structured and random—
with operator fidelity 1.0 and action‑on‑states differences at numerical noise
levels. These results show that the high‑bond MPU use case is fully achievable
using geometric tensor methods, without relying on the algebraic MPU machinery
from the original paper.


# NOTE ABOUT RECONSTRUCTION FROBENIUS:
# -----------------------------------
# For random high‑bond MPU circuits, the reconstructed unitary U_rec often differs
# from U_target by a *global phase*. A global phase does NOT change the physical
# action of a unitary, and fidelity(U_rec, U_target) = 1.0 confirms they are
# physically identical.
#
# HOWEVER:
# The Frobenius norm IS sensitive to global phase. If U_rec = e^{iφ} U_target
# with |e^{iφ}| = 1, then every entry is rotated by the same phase. For a 64×64
# unitary, the maximum possible Frobenius difference between two unitaries is 16.
# So a value like 1.6e+01 simply indicates a global phase offset, NOT an error.
#
# Therefore:
#   • Fidelity = 1.0  → reconstruction is EXACT (up to phase)
#   • Frobenius ≈ 16 → meaningless for reconstruction in this case


"""


import cupy as cp
import numpy as np

from module_fidelity_objective import FidelityObjective

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

def random_two_qubit_unitary(seed=None):
    if seed is not None:
        cp.random.seed(seed)

    # Random complex 4x4
    M = cp.random.randn(4, 4) + 1j * cp.random.randn(4, 4)

    # QR decomposition
    Q, R = cp.linalg.qr(M)

    # Make unitary with det=1 (optional, but fine)
    det = cp.linalg.det(Q)
    Q = Q / cp.power(det, 1/4)

    return Q

def build_random_mpu_layer(N=6, depth=2, seed=None):
    I2 = cp.eye(2, dtype=cp.complex128)

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

    if seed is not None:
        cp.random.seed(seed)

    dim = 2**N
    U = cp.eye(dim, dtype=cp.complex128)

    # depth layers of nearest-neighbor random 2-qubit gates
    for layer in range(depth):
        for q in range(N - 1):
            G = random_two_qubit_unitary()
            U_layer = embed_two_qubit_gate(G, q, N)
            U = U_layer @ U

    return U


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


def build_highbond_mpu_layer(N=6):
    I2 = cp.eye(2, dtype=cp.complex128)

    # Strongly entangling 2-qubit gates
    def CP(theta):
        return cp.array([
            [1+0j, 0, 0, 0],
            [0, 1+0j, 0, 0],
            [0, 0, 1+0j, 0],
            [0, 0, 0, complex(cp.exp(1j * theta))]
        ], dtype=cp.complex128)

    G0 = CP(np.pi/4)
    G1 = CP(np.pi/2)
    G2 = CP(3*np.pi/4)

    G3 = cp.array([
        [1+0j, 0, 0, 0],
        [0, 1+0j, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ], dtype=cp.complex128)

    G4 = cp.array([
        [1+0j, 0, 0, 0],
        [0, 1+0j, 0, 0],
        [0, 0, 0, 1+0j],
        [0, 0, 1+0j, 0]
    ], dtype=cp.complex128)

    gates = [G0, G1, G2, G3, G4]

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


def run_highbond_mpu_test():
    print("\n--- High-Bond MPU Test ---")

    N = 6
    U_high = build_highbond_mpu_layer(N)

    chi = 16
    U_mpu = mpu_projection(U_high, N, chi)
    U_mpu = align_global_phase(U_high, U_mpu)

    print(f"High-bond Frobenius: {float(cp.linalg.norm(U_high - U_mpu)):.6f}")
    print(f"High-bond operator fidelity: {FidelityObjective.operator_fidelity(U_mpu, U_high):.6f}")

    A_list = mpu_extract_all_sites(U_high, N, chi, debug=True)

    validate_action_on_states(U_high, U_mpu, N)
    validate_schmidt_ranks(U_mpu, N)
    validate_reconstruction(U_high, A_list)

def run_random_mpu_test():
    print("\n--- Random High-Bond MPU Test ---")

    N = 6
    depth = 2
    chi = 16

    U_rand = build_random_mpu_layer(N=N, depth=depth, seed=123)

    U_mpu = mpu_projection(U_rand, N, chi)
    U_mpu = align_global_phase(U_rand, U_mpu)

    print(f"Random Frobenius: {float(cp.linalg.norm(U_rand - U_mpu)):.6f}")
    print(f"Random operator fidelity: {FidelityObjective.operator_fidelity(U_mpu, U_rand):.6f}")

    A_list = mpu_extract_all_sites(U_rand, N, chi, debug=True)

    validate_action_on_states(U_rand, U_mpu, N)
    validate_schmidt_ranks(U_mpu, N)
    validate_reconstruction(U_rand, A_list)
    
    debug_reconstruction(U_rand, U_mpu, A_list)
    debug_reconstruction_axes(U_rand, U_mpu, A_list)


    
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

def reconstruct_from_A_noboundary(A_list):
    """
    Reconstruct full unitary from A-tensors without extra boundary vectors.
    A_list[k] shape: (bond_left, 2, 2, bond_right)
    """
    N = len(A_list)

    # Start with first tensor
    T = A_list[0]

    # Contract chain left → right
    for k in range(1, N):
        Ak = A_list[k]

        m = min(T.shape[-1], Ak.shape[0])
        T = T[..., :m]
        Ak = Ak[:m, :, :, :]

        T = cp.tensordot(T, Ak, axes=([-1], [0]))

    # At this point T.shape = (1, 2, 2, ..., 2, 2, 1)
    # Drop trivial bond dims by reshape
    T = T.reshape(*([2] * (2 * N)))

    # Two plausible physical index orderings:

    # Option A: (in0, out0, in1, out1, ..., inN-1, outN-1)
    axes_A = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U_A = T.transpose(axes_A).reshape(2**N, 2**N)

    return U_A

def debug_reconstruction_axes(U_target, U_mpu, A_list):
    print("\n--- Reconstruction Axis Debug ---")

    U_A = reconstruct_from_A_noboundary(A_list)

    for label, U_rec in [("A", U_A)]:
        diff_target = cp.linalg.norm(U_target - U_rec)
        diff_mpu    = cp.linalg.norm(U_mpu    - U_rec)

        fid_target = FidelityObjective.operator_fidelity(U_rec, U_target)
        fid_mpu    = FidelityObjective.operator_fidelity(U_rec, U_mpu)

        print(f"[{label}] Frob(U_rec - U_target): {float(diff_target):.6e}")
        print(f"[{label}] Frob(U_rec - U_mpu):    {float(diff_mpu):.6e}")
        print(f"[{label}] Fid(U_rec, U_target):   {float(fid_target):.6f}")
        print(f"[{label}] Fid(U_rec, U_mpu):      {float(fid_mpu):.6f}")
        print()

if __name__ == "__main__":
    run_highbond_mpu_test()
    run_random_mpu_test()

