"""
David Mulnix copyright 2026

This script implements a multiscale block-collapse flow on the unitary group U(n),
using a closed-form block energy derived from David’s multiscale spectral algebra.

High-level idea
---------------
We start with two unitary matrices:
    • U_ref ∈ U(n): a random “reference” unitary.
    • U ∈ U(n): a random “initial” unitary that will be evolved.

The goal is not to reproduce U_ref exactly, but to move U along U(n) so that its
multiscale block structure (in David’s algebraic sense) becomes closer to that of U_ref.
The script does this by defining a block-based energy and then performing a constrained
descent on U while re-projecting back to U(n) at every step.

Random unitary generation
-------------------------
def random_unitary(n):
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    return Q

Here, X is a dense complex Gaussian matrix. QR decomposition produces Q with
orthonormal columns, and in the complex setting this gives a unitary matrix.
This works for any n (4, 8, 12, 16, 32, …); the default n=4 is just a convenience.
When you change n in main(), random_unitary(n) automatically generates an n×n unitary.

Multiscale block structure
--------------------------
def multiscale_blocks(n):
    blocks = []
    k = int(np.log2(n))
    for j in range(1, k+1):
        size = 2**j
        count = n // size
        for r in range(count):
            for c in range(count):
                rs = r * size
                re = rs + size
                cs = c * size
                ce = cs + size
                blocks.append((rs, re, cs, ce, size))
    return blocks

This function partitions the n×n matrix into dyadic blocks:
    • At scale j, block size = 2^j.
    • The matrix is tiled into (n / 2^j) × (n / 2^j) blocks of that size.
These blocks are the multiscale “windows” through which the algebra looks at U.

def multiscale_block_sums(U):
    blocks = multiscale_blocks(U.shape[0])
    sums = []
    for (r0, r1, c0, c1, size) in blocks:
        B = U[r0:r1, c0:c1]
        s = np.sum(B)
        sums.append(s)
    return np.array(sums, dtype=complex)

For each block B, we compute the complex block-sum s(B) = Σ_{i,j in B} U_{ij}.
The collection of all block-sums across all scales is a multiscale feature vector
for U in David’s spectral algebra.

Block energy
------------
def block_energy(U, U_ref):
    s_U  = multiscale_block_sums(U)
    s_ref = multiscale_block_sums(U_ref)
    diff = s_U - s_ref
    return np.sum(np.abs(diff)**2)

This is the closed-form multiscale energy:
    E(U, U_ref) = Σ_blocks |s_U(B) - s_ref(B)|^2.
It measures how different U is from U_ref in the multiscale block algebra, not
entrywise, but through aggregated spectral-like block features.

Coordinate descent on U(n)
--------------------------
def coordinate_descent_step(U, U_ref, step_size=0.1):
    n = U.shape[0]
    U_work = U.copy()
    E_current = block_energy(U_work, U_ref)

    for i in range(n):
        for j in range(n):
            # small complex perturbation
            delta = step_size * (np.random.randn() + 1j * np.random.randn())
            U_candidate = U_work.copy()
            U_candidate[i, j] += delta

            # re-unitarize
            Q, _ = np.linalg.qr(U_candidate)
            E_candidate = block_energy(Q, U_ref)

            if E_candidate < E_current:
                U_work = Q
                E_current = E_candidate

    return U_work, E_current

For each entry U[i,j], we:
    • Add a small random complex perturbation δ.
    • Project back to U(n) via QR: Q is unitary again.
    • Evaluate the block energy E(Q, U_ref).
    • Accept Q only if it strictly lowers the energy.

This is a monotone coordinate descent in the block energy, constrained to U(n)
by QR projection. The flow is “collapse-like” in the sense that U is driven
toward a lower-energy multiscale configuration relative to U_ref.

Structured collapse flow and diagnostics
----------------------------------------
def run_structured_collapse(n=4, steps=100, step_size=0.1):
    U_ref = random_unitary(n)
    U = random_unitary(n)

    E0 = block_energy(U, U_ref)
    print(f"Initial block energy: {E0:.6f}")

    best_E = E0
    best_U = U.copy()

    for t in range(steps):
        U, E_current = coordinate_descent_step(U, U_ref, step_size=step_size)

        if E_current < best_E:
            best_E = E_current
            best_U = U.copy()

        if t % 10 == 0:
            print(f"step {t:3d} | current_E={E_current:.6f} | best_E={best_E:.6f}")

    print("\nFinal best energy:", best_E)
    print("\nFinal unitary U_final:")
    print(best_U)

    # Unitarity check
    UUdag = best_U @ best_U.conj().T
    print("\nU_final * U_final^†:")
    print(UUdag)
    print("\nIs U_final unitary (allclose to I)?",
          np.allclose(UUdag, np.eye(n), atol=1e-8))

    # Eigenvalue check
    eigvals = np.linalg.eigvals(best_U)
    print("\nEigenvalues of U_final:")
    print(eigvals)
    print("\nEigenvalue magnitudes (should all be 1):")
    print(np.abs(eigvals))

    # Determinant check
    detU = np.linalg.det(best_U)
    print("\nDeterminant of U_final:")
    print(detU)
    print("Determinant magnitude (should be 1):", np.abs(detU))

    # Spectral radius check
    spectral_radius = max(np.abs(eigvals))
    print("\nSpectral radius (should be 1):", spectral_radius)

    # Frobenius distance
    frob_dist = np.linalg.norm(best_U - U_ref, 'fro')
    print("\nFrobenius distance ||U_final - U_ref||_F:")
    print(frob_dist)

    # Action on computational basis
    print("\nAction of U_final on computational basis states:")
    for k in range(n):
        basis = np.zeros((n,), dtype=complex)
        basis[k] = 1.0
        transformed = best_U @ basis
        print(f"|{k}>  →  {transformed}")

    return best_U, U_ref

The diagnostics confirm that:
    • U_final is unitary: U_final U_final† ≈ I_n.
    • All eigenvalues lie on the unit circle.
    • |det(U_final)| = 1 and spectral radius = 1.
    • U_final is generally different from U_ref (nonzero Frobenius distance),
      but it is a fully valid unitary that has been shaped by the multiscale
      block energy rather than random noise.

Main entry point
----------------
def main():
    np.random.seed(0)
    run_structured_collapse(n=4, steps=100, step_size=0.1)
    # validated for n = 4, 8, 12, 16; can be extended to n = 32, 64, ...

if __name__ == "__main__":
    main()

Changing n in main() simply changes the dimension of U_ref and U; random_unitary(n)
and the block machinery adapt automatically. The collapse flow then operates on U(n)
for that dimension, producing a new unitary U_final that is:
    • distinct from U_ref,
    • fully unitary,
    • and optimized with respect to David’s multiscale spectral block energy.

Conceptually, this is “Multiscale Spectral Algebra: Unitary Edition”: the same
closed-form multiscale algebra that was validated on Hadamard matrices is now
transplanted into the unitary group, where it defines a geometric energy and a
collapse flow on quantum operators.
"""



import numpy as np

# ---------- pure unitary, n=4 ----------

def random_unitary(n=4):
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    return Q

# ---------- your multiscale blocks (n=4: blocks of size 2 and 4) ----------

def multiscale_blocks(n):
    blocks = []
    k = int(np.log2(n))
    for j in range(1, k+1):
        size = 2**j
        count = n // size
        for r in range(count):
            for c in range(count):
                rs = r * size
                re = rs + size
                cs = c * size
                ce = cs + size
                blocks.append((rs, re, cs, ce, size))
    return blocks

def multiscale_block_sums(U):
    blocks = multiscale_blocks(U.shape[0])
    sums = []
    for (r0, r1, c0, c1, size) in blocks:
        B = U[r0:r1, c0:c1]
        s = np.sum(B)
        sums.append(s)
    return np.array(sums, dtype=complex)

# ---------- energy: how far U is from U_ref in your block algebra ----------

def block_energy(U, U_ref):
    s_U  = multiscale_block_sums(U)
    s_ref = multiscale_block_sums(U_ref)
    diff = s_U - s_ref
    return np.sum(np.abs(diff)**2)

# ---------- coordinate descent step on unitary ----------

def coordinate_descent_step(U, U_ref, step_size=0.1):
    """
    Simple coordinate descent:
    - for each entry U[i,j], try a small complex perturbation
    - keep the change if it lowers block_energy
    - re-unitarize at the end
    """
    n = U.shape[0]
    U_work = U.copy()
    E_current = block_energy(U_work, U_ref)

    for i in range(n):
        for j in range(n):
            # small complex perturbation
            delta = step_size * (np.random.randn() + 1j * np.random.randn())
            U_candidate = U_work.copy()
            U_candidate[i, j] += delta

            # re-unitarize
            Q, _ = np.linalg.qr(U_candidate)
            E_candidate = block_energy(Q, U_ref)

            if E_candidate < E_current:
                U_work = Q
                E_current = E_candidate

    return U_work, E_current

# ---------- structured collapse flow ----------

def run_structured_collapse(n=4, steps=100, step_size=0.1):
    U_ref = random_unitary(n)
    U = random_unitary(n)

    E0 = block_energy(U, U_ref)
    print(f"Initial block energy: {E0:.6f}")

    best_E = E0
    best_U = U.copy()

    for t in range(steps):
        U, E_current = coordinate_descent_step(U, U_ref, step_size=step_size)

        if E_current < best_E:
            best_E = E_current
            best_U = U.copy()

        if t % 10 == 0:
            print(f"step {t:3d} | current_E={E_current:.6f} | best_E={best_E:.6f}")

    #print("\nFinal best energy:", best_E)


    print("\nFinal best energy:", best_E)
    print("\nFinal unitary U_final:")
    print(best_U)

    # Unitarity check
    UUdag = best_U @ best_U.conj().T
    print("\nU_final * U_final^†:")
    print(UUdag)
    print("\nIs U_final unitary (allclose to I)?",
          np.allclose(UUdag, np.eye(n), atol=1e-8))

    # Eigenvalue check
    eigvals = np.linalg.eigvals(best_U)
    print("\nEigenvalues of U_final:")
    print(eigvals)
    print("\nEigenvalue magnitudes (should all be 1):")
    print(np.abs(eigvals))

    # Determinant check
    detU = np.linalg.det(best_U)
    print("\nDeterminant of U_final:")
    print(detU)
    print("Determinant magnitude (should be 1):", np.abs(detU))

    # Spectral radius check
    spectral_radius = max(np.abs(eigvals))
    print("\nSpectral radius (should be 1):", spectral_radius)


    # ---------------------------------------------------------
    # Frobenius distance between U_final and U_ref
    # ---------------------------------------------------------
    frob_dist = np.linalg.norm(best_U - U_ref, 'fro')
    print("\nFrobenius distance ||U_final - U_ref||_F:")
    print(frob_dist)

    # ---------------------------------------------------------
    # Apply U_final to basis states |0>, |1>, |2>, |3>
    # ---------------------------------------------------------
    print("\nAction of U_final on computational basis states:")

    for k in range(n):
        basis = np.zeros((n,), dtype=complex)
        basis[k] = 1.0
        transformed = best_U @ basis
        print(f"|{k}>  →  {transformed}")

    return best_U, U_ref

# ---------- MAIN ----------

def main():
    np.random.seed(0)
    run_structured_collapse(n=4, steps=100, step_size=0.1) # this has been validated on in 4, 8, 12, 16 

if __name__ == "__main__":
    main()


