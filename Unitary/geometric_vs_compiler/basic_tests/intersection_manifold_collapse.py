"""
David Mulnix copyright 2026


Intersection Manifold Collapse (Geometric Unitary Experiment)

This script performs collapse on *intersection manifolds* inside U(N), combining
two of the previously defined geometric unitary manifolds into a single energy
functional. Instead of collapsing toward one structure (tensor, diagonal,
block‑diagonal, stabilizer, or hardware‑efficient), this script collapses toward
a weighted combination of two manifolds simultaneously.

Supported manifold energies (imported from other scripts):
• diagonal_energy(U)
• multifactor_tensor_energy(U, factors)
• block_diagonal_energy(U, block_sizes)
• stabilizer_energy(U, pauli_subset)
• hardware_efficient_energy(U, U_he)

Intersection energies take the form:
    E_intersection(U) = α * E_A(U) + (1 − α) * E_B(U)

Examples implemented here:
• diagonal + multifactor tensor
• hardware‑efficient + stabilizer
• hardware‑efficient + multifactor tensor
• hardware‑efficient + block‑diagonal

The collapse engine:
• Generates a random unitary U using QR.
• Applies random perturbations and re‑unitarizes via QR.
• Evaluates the intersection energy.
• Accepts updates only when energy decreases.
• Uses step‑size growth/decay and early stopping.
• Verifies unitarity at the end.

This script acts as a *meta‑collapse controller* that uses all previously defined
manifold energies to explore hybrid geometric structures inside U(N).


======================================
Running intersection: diagonal + multifactor tensor
======================================
[n=32, diag+tensor] Initial energy: 58.987380
[diag+tensor] step   0 | current_E=56.045902 | best_E=56.045902 | rel_improve=4.987e-02 | step_size=8.800e-02
[diag+tensor] step   5 | current_E=55.274385 | best_E=55.274385 | rel_improve=1.066e-02 | step_size=3.652e-02
[n=32, diag+tensor] Early stopping at step 7 (rel_improve=0.000e+00)

[n=32, diag+tensor] Final best energy: 53.082344
[n=32, diag+tensor] Total time: 0.999 s (steps_run=8)
[n=32, diag+tensor] Acceptance ratio: 0.500
[n=32, diag+tensor] Is unitary: True

=== FULL UNITARITY AUDIT: diag+tensor_n32 ===
Direct unitarity: True
Column orthonormality: True
Det magnitude: 1.000000000000 (ok=True)
Eigenvalues on unit circle: True
{'n': 32, 'label': 'diag+tensor', 'steps_configured': 40, 'steps_run': 8, 'num_candidates': 16, 'initial_energy': 58.98737991093492, 'final_energy': 53.08234392578922, 'total_time': 0.9993994235992432, 'avg_iter_time': 0.12435966730117798, 'accept_ratio': 0.5, 'is_unitary': True}

======================================
Running intersection: hardware + stabilizer
======================================
[n=8, hardware+stabilizer] Initial energy: 5.391225
[hardware+stabilizer] step   0 | current_E=3.612014 | best_E=3.612014 | rel_improve=3.300e-01 | step_size=8.800e-02
[hardware+stabilizer] step   5 | current_E=5.031848 | best_E=3.612014 | rel_improve=0.000e+00 | step_size=1.479e-02
[n=8, hardware+stabilizer] Early stopping at step 6 (rel_improve=0.000e+00)

[n=8, hardware+stabilizer] Final best energy: 3.612014
[n=8, hardware+stabilizer] Total time: 2.204 s (steps_run=7)
[n=8, hardware+stabilizer] Acceptance ratio: 1.000
[n=8, hardware+stabilizer] Is unitary: True

=== FULL UNITARITY AUDIT: hardware+stabilizer_n8 ===
Direct unitarity: True
Column orthonormality: True
Det magnitude: 1.000000000000 (ok=True)
Eigenvalues on unit circle: True
{'n': 8, 'label': 'hardware+stabilizer', 'steps_configured': 40, 'steps_run': 7, 'num_candidates': 8, 'initial_energy': 5.391224930963892, 'final_energy': 3.612013898699115, 'total_time': 2.2037582397460938, 'avg_iter_time': 0.3148226056780134, 'accept_ratio': 1.0, 'is_unitary': True}

======================================
Running intersection: hardware + multifactor tensor
======================================
[n=8, hardware+tensor] Initial energy: 16.556300
[hardware+tensor] step   0 | current_E=13.197044 | best_E=13.197044 | rel_improve=2.029e-01 | step_size=8.800e-02
[hardware+tensor] step   5 | current_E=11.639937 | best_E=10.967819 | rel_improve=0.000e+00 | step_size=3.652e-02
[n=8, hardware+tensor] Early stopping at step 6 (rel_improve=0.000e+00)

[n=8, hardware+tensor] Final best energy: 10.967819
[n=8, hardware+tensor] Total time: 0.554 s (steps_run=7)
[n=8, hardware+tensor] Acceptance ratio: 0.857
[n=8, hardware+tensor] Is unitary: True

=== FULL UNITARITY AUDIT: hardware+tensor_n8 ===
Direct unitarity: True
Column orthonormality: True
Det magnitude: 1.000000000000 (ok=True)
Eigenvalues on unit circle: True
{'n': 8, 'label': 'hardware+tensor', 'steps_configured': 40, 'steps_run': 7, 'num_candidates': 8, 'initial_energy': 16.55629993960117, 'final_energy': 10.967819114343525, 'total_time': 0.554171085357666, 'avg_iter_time': 0.07902414458138603, 'accept_ratio': 0.8571428571428571, 'is_unitary': True}

======================================
Running intersection: hardware + block
======================================
[n=16, hardware+block] Initial energy: 30.766334
[hardware+block] step   0 | current_E=21.818938 | best_E=21.818938 | rel_improve=2.908e-01 | step_size=8.800e-02
[hardware+block] step   5 | current_E=22.532571 | best_E=20.999319 | rel_improve=0.000e+00 | step_size=5.739e-02
[n=16, hardware+block] Early stopping at step 6 (rel_improve=0.000e+00)

[n=16, hardware+block] Final best energy: 20.999319
[n=16, hardware+block] Total time: 0.747 s (steps_run=7)
[n=16, hardware+block] Acceptance ratio: 0.714
[n=16, hardware+block] Is unitary: True

=== FULL UNITARITY AUDIT: hardware+block_n16 ===
Direct unitarity: True
Column orthonormality: True
Det magnitude: 1.000000000000 (ok=True)
Eigenvalues on unit circle: True
{'n': 16, 'label': 'hardware+block', 'steps_configured': 40, 'steps_run': 7, 'num_candidates': 8, 'initial_energy': 30.766334388120086, 'final_energy': 20.999318846076477, 'total_time': 0.747025728225708, 'avg_iter_time': 0.10657501220703125, 'accept_ratio': 0.7142857142857143, 'is_unitary': True}

"""

import cupy as cp
import numpy as np
import time

# === stabilizer + Pauli tools ===
from block_stabilizer_collapse import (
    stabilizer_energy,
    generate_pauli_basis
)

# === hardware-efficient tools ===
from hardware_efficient_collapse import (
    hardware_efficient_projection,
    hardware_efficient_energy
)

# === block-diagonal manifold ===
from block_diagonal_collapse import (
    block_diagonal_energy
)

# === diagonal manifold ===
from diagnoal_unitary_collapse import (
    diagonal_energy
)

# === tensor / multifactor manifold ===
from geometric_multifactor_tensor_validation import (
    multifactor_tensor_energy
)


# ============================================================
# ASSUMPTIONS: you already have these functions implemented:
# ============================================================
# - tensor_energy(U, tensor_config)
# - block_diagonal_energy(U, block_sizes) -> (E_bd, B)
# - diagonal_energy(U)  # or equivalent diagonal manifold energy
# - stabilizer_energy(U, pauli_subset)
# - hardware_efficient_projection(n_qubits, depth) -> U_he
# - hardware_efficient_energy(U, U_he)
#
# If names differ, just swap them in below.

# ============================================================
# Combined / intersection energies (UPDATED FOR REAL IMPORTS)
# ============================================================

def energy_diag_tensor(U, factors, alpha=0.5):
    """
    Intersection: diagonal + multifactor tensor manifold.
    E(U) = alpha * E_diag(U) + (1-alpha) * E_tensor(U)
    """
    # diagonal_energy returns (E_diag, D)
    E_diag, _ = diagonal_energy(U)

    # multifactor_tensor_energy returns (E_tensor, U_tensor, G_list)
    E_tensor, _, _ = multifactor_tensor_energy(U, factors)

    return alpha * E_diag + (1.0 - alpha) * E_tensor


def energy_hardware_stabilizer(U, n_qubits, depth, pauli_subset, alpha=0.5):
    """
    Intersection: hardware-efficient + stabilizer/Clifford-like manifold.
    """
    U_he = hardware_efficient_projection(n_qubits, depth=depth)
    E_he = hardware_efficient_energy(U, U_he)

    E_stab = stabilizer_energy(U, pauli_subset)

    return alpha * E_stab + (1.0 - alpha) * E_he


def energy_hardware_tensor(U, n_qubits, depth, factors, alpha=0.5):
    """
    Intersection: hardware-efficient + multifactor tensor manifold.
    """
    U_he = hardware_efficient_projection(n_qubits, depth=depth)
    E_he = hardware_efficient_energy(U, U_he)

    E_tensor, _, _ = multifactor_tensor_energy(U, factors)

    return alpha * E_he + (1.0 - alpha) * E_tensor


def energy_hardware_block(U, n_qubits, depth, block_sizes, alpha=0.5):
    """
    Intersection: hardware-efficient + block-diagonal manifold.
    """
    U_he = hardware_efficient_projection(n_qubits, depth=depth)
    E_he = hardware_efficient_energy(U, U_he)
    E_bd, _ = block_diagonal_energy(U, block_sizes)
    return alpha * E_bd + (1.0 - alpha) * E_he

# ============================================================
# Generic intersection collapse flow
# ============================================================

def random_unitary(n):
    X = cp.random.randn(n, n) + 1j * cp.random.randn(n, n)
    Q, _ = cp.linalg.qr(X)
    return Q

def batch_descent_step_intersection(U, energy_fn, step_size=0.1, num_candidates=8, energy_args=None):
    if energy_args is None:
        energy_args = {}

    E_current = energy_fn(U, **energy_args)
    best_U = U
    best_E = E_current
    accepted = False

    n = U.shape[0]

    for _ in range(num_candidates):
        U_candidate = U.copy()

        delta = step_size * (cp.random.randn(n, n) + 1j * cp.random.randn(n, n))
        U_candidate = U_candidate + delta

        Q, _ = cp.linalg.qr(U_candidate)

        E_candidate = energy_fn(Q, **energy_args)

        if E_candidate < best_E:
            best_E = E_candidate
            best_U = Q
            accepted = True

    return best_U, best_E, accepted

# =========================
# Full Unitarity Audit (Intersection Manifold)
# =========================

def full_unitarity_audit(U, name="U"):
    """
    Performs a comprehensive unitarity audit on the matrix U.
    This verifies that the collapse result is a true unitary and
    saves the matrix for external inspection.
    """
    U_cpu = cp.asnumpy(U)

    # 1. Direct unitarity check: U U† = I
    direct = np.allclose(U_cpu @ U_cpu.conj().T, np.eye(U_cpu.shape[0]), atol=1e-12)

    # 2. Column orthonormality check
    cols = U_cpu.T
    norms = [np.linalg.norm(c) for c in cols]
    orthonormal = all(abs(n - 1) < 1e-12 for n in norms)

    # 3. Determinant magnitude check
    det_mag = abs(np.linalg.det(U_cpu))
    det_ok = abs(det_mag - 1) < 1e-12

    # 4. Eigenvalues on the unit circle
    eigs = np.linalg.eigvals(U_cpu)
    eig_ok = all(abs(abs(ev) - 1) < 1e-12 for ev in eigs)

    print(f"\n=== FULL UNITARITY AUDIT: {name} ===")
    print(f"Direct unitarity: {direct}")
    print(f"Column orthonormality: {orthonormal}")
    print(f"Det magnitude: {det_mag:.12f} (ok={det_ok})")
    print(f"Eigenvalues on unit circle: {eig_ok}")

    # Save matrix for external inspection
    np.save(f"unitarity_audit_{name}.npy", U_cpu)


def run_intersection_collapse(
    n,
    energy_fn,
    energy_args=None,
    steps=40,
    step_size_init=0.08,
    num_candidates=8,
    tol=1e-3,
    step_decay=0.7,
    step_growth=1.1,
    label="intersection"
):
    if energy_args is None:
        energy_args = {}

    U = random_unitary(n)

    E0 = energy_fn(U, **energy_args)
    print(f"[n={n}, {label}] Initial energy: {float(E0):.6f}")

    best_E = E0
    best_U = U.copy()

    energies = []
    times = []
    accepts = 0

    step_size = step_size_init
    start_total = time.time()

    for t in range(steps):
        iter_start = time.time()
        U, E_current, accepted = batch_descent_step_intersection(
            U,
            energy_fn,
            step_size=step_size,
            num_candidates=num_candidates,
            energy_args=energy_args
        )
        iter_end = time.time()

        if accepted:
            accepts += 1

        if E_current < best_E:
            rel_improve = float((best_E - E_current) / (best_E + 1e-12))
            best_E = E_current
            best_U = U.copy()
            step_size *= step_growth
        else:
            rel_improve = 0.0
            step_size *= step_decay

        energies.append(float(E_current))
        times.append(iter_end - iter_start)

        if t % 5 == 0:
            print(
                f"[{label}] step {t:3d} | current_E={float(E_current):.6f} | "
                f"best_E={float(best_E):.6f} | rel_improve={rel_improve:.3e} | step_size={step_size:.3e}"
            )

        if rel_improve < tol and t > 5:
            print(f"[n={n}, {label}] Early stopping at step {t} (rel_improve={rel_improve:.3e})")
            break

    end_total = time.time()
    total_time = end_total - start_total
    steps_run = len(energies)
    accept_ratio = accepts / steps_run if steps_run > 0 else 0.0

    UUdag = best_U @ best_U.conj().T
    is_unitary = cp.allclose(UUdag, cp.eye(n), atol=1e-8)

    print(f"\n[n={n}, {label}] Final best energy: {float(best_E):.6f}")
    print(f"[n={n}, {label}] Total time: {total_time:.3f} s (steps_run={steps_run})")
    print(f"[n={n}, {label}] Acceptance ratio: {accept_ratio:.3f}")
    print(f"[n={n}, {label}] Is unitary: {bool(is_unitary)}")

    metrics = {
        "n": n,
        "label": label,
        "steps_configured": steps,
        "steps_run": steps_run,
        "num_candidates": num_candidates,
        "initial_energy": float(E0),
        "final_energy": float(best_E),
        "total_time": total_time,
        "avg_iter_time": sum(times) / len(times),
        "accept_ratio": accept_ratio,
        "is_unitary": bool(is_unitary),
    }

    #Audit Matrix
    full_unitarity_audit(best_U, name=f"{label}_n{n}")

    return metrics

# ============================================================
# Specific intersection tests (UPDATED, NO PLACEHOLDERS)
# ============================================================

def intersection_benchmark():
    # 1) diagonal + tensor (multifactor)
    factors_diag = [2, 2, 8]  # n = 32
    print("\n======================================")
    print("Running intersection: diagonal + multifactor tensor")
    print("======================================")
    metrics_diag_tensor = run_intersection_collapse(
        n=32,
        energy_fn=energy_diag_tensor,
        energy_args={
            "factors": factors_diag,
            "alpha": 0.5
        },
        steps=40,
        step_size_init=0.08,
        num_candidates=16,
        label="diag+tensor"
    )
    print(metrics_diag_tensor)

    # 2) hardware + stabilizer
    print("\n======================================")
    print("Running intersection: hardware + stabilizer")
    print("======================================")
    n_qubits_hs = 3
    N_hs = 2**n_qubits_hs
    pauli_subset = generate_pauli_basis(n_qubits_hs)[:16]
    metrics_hardware_stab = run_intersection_collapse(
        n=N_hs,
        energy_fn=energy_hardware_stabilizer,
        energy_args={
            "n_qubits": n_qubits_hs,
            "depth": 3,
            "pauli_subset": pauli_subset,
            "alpha": 0.5
        },
        steps=40,
        step_size_init=0.08,
        num_candidates=8,
        label="hardware+stabilizer"
    )
    print(metrics_hardware_stab)

    # 3) hardware + tensor (multifactor)
    print("\n======================================")
    print("Running intersection: hardware + multifactor tensor")
    print("======================================")
    n_qubits_ht = 3
    N_ht = 2**n_qubits_ht
    factors_ht = [2, 2, 2]  # n = 8
    metrics_hardware_tensor = run_intersection_collapse(
        n=N_ht,
        energy_fn=energy_hardware_tensor,
        energy_args={
            "n_qubits": n_qubits_ht,
            "depth": 3,
            "factors": factors_ht,
            "alpha": 0.5
        },
        steps=40,
        step_size_init=0.08,
        num_candidates=8,
        label="hardware+tensor"
    )
    print(metrics_hardware_tensor)

    # 4) hardware + block
    print("\n======================================")
    print("Running intersection: hardware + block")
    print("======================================")
    n_qubits_hb = 4
    N_hb = 2**n_qubits_hb
    block_sizes_hb = [8, 8]  # N_hb = 16
    metrics_hardware_block = run_intersection_collapse(
        n=N_hb,
        energy_fn=energy_hardware_block,
        energy_args={
            "n_qubits": n_qubits_hb,
            "depth": 4,
            "block_sizes": block_sizes_hb,
            "alpha": 0.5
        },
        steps=40,
        step_size_init=0.08,
        num_candidates=8,
        label="hardware+block"
    )
    print(metrics_hardware_block)

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    cp.random.seed(0)
    intersection_benchmark()
