"""

David Mulnix copyright 2026

This is an optimized version of the original script that ran up to n 16, it runs on a single GPU and produces the following result. 

======================================
Running structured speed-optimized collapse for n = 32
======================================
[n=32] Initial block energy: 238.389831
step   0 | current_E=227.871754 | best_E=227.871754 | rel_improve=4.412e-02 | step_size=8.800e-02
step   5 | current_E=199.813100 | best_E=199.813100 | rel_improve=4.732e-02 | step_size=1.417e-01
step  10 | current_E=162.176354 | best_E=162.176354 | rel_improve=4.263e-02 | step_size=2.282e-01
step  15 | current_E=153.908438 | best_E=153.908438 | rel_improve=8.048e-03 | step_size=3.676e-01
step  20 | current_E=141.556473 | best_E=141.556473 | rel_improve=1.429e-02 | step_size=5.920e-01
[n=32] Early stopping at step 21 (rel_improve=5.533e-04)

[n=32] Final best energy: 141.478153
[n=32] Total time: 3.383 s (steps_run=22)
[n=32] Acceptance ratio: 1.000
[n=32] Is unitary: True
[n=32] Det magnitude: 1.000000
[n=32] Frobenius distance: 7.779574

======================================
Running structured speed-optimized collapse for n = 64
======================================
[n=64] Initial block energy: 599.305368
step   0 | current_E=599.305368 | best_E=599.305368 | rel_improve=0.000e+00 | step_size=5.600e-02
step   5 | current_E=566.880567 | best_E=566.880567 | rel_improve=2.201e-02 | step_size=9.019e-02
step  10 | current_E=538.243655 | best_E=538.243655 | rel_improve=1.082e-02 | step_size=1.452e-01
[n=64] Early stopping at step 13 (rel_improve=0.000e+00)

[n=64] Final best energy: 532.484154
[n=64] Total time: 7.355 s (steps_run=14)
[n=64] Acceptance ratio: 0.857
[n=64] Is unitary: True
[n=64] Det magnitude: 1.000000
[n=64] Frobenius distance: 11.255230

======================================
Running structured speed-optimized collapse for n = 128
======================================
[n=128] Initial block energy: 1538.122167
step   0 | current_E=1469.933967 | best_E=1469.933967 | rel_improve=4.433e-02 | step_size=8.800e-02
step   5 | current_E=1218.254938 | best_E=1218.254938 | rel_improve=4.002e-03 | step_size=1.417e-01
step  10 | current_E=1195.962236 | best_E=1195.962236 | rel_improve=0.000e+00 | step_size=1.452e-01
[n=128] Early stopping at step 10 (rel_improve=0.000e+00)

[n=128] Final best energy: 1195.962236
[n=128] Total time: 20.314 s (steps_run=11)
[n=128] Acceptance ratio: 0.909
[n=128] Is unitary: True
[n=128] Det magnitude: 1.000000
[n=128] Frobenius distance: 15.985651

======================================
Running structured speed-optimized collapse for n = 256
======================================
[n=256] Initial block energy: 4746.212435
step   0 | current_E=3920.048502 | best_E=3920.048502 | rel_improve=1.741e-01 | step_size=8.800e-02
step   5 | current_E=3536.731672 | best_E=3536.731672 | rel_improve=6.613e-04 | step_size=1.417e-01
[n=256] Early stopping at step 7 (rel_improve=3.684e-04)

[n=256] Final best energy: 3530.970823
[n=256] Total time: 56.276 s (steps_run=8)
[n=256] Acceptance ratio: 1.000
[n=256] Is unitary: True
[n=256] Det magnitude: 1.000000
[n=256] Frobenius distance: 22.630540

======================================
Running structured speed-optimized collapse for n = 1024
======================================
[n=1024] Initial block energy: 20372.412291
step   0 | current_E=18285.806517 | best_E=18285.806517 | rel_improve=1.024e-01 | step_size=8.800e-02
step   5 | current_E=17449.770101 | best_E=17449.770101 | rel_improve=7.909e-03 | step_size=1.417e-01
[n=1024] Early stopping at step 6 (rel_improve=3.484e-04)

[n=1024] Final best energy: 17443.691248
[n=1024] Total time: 1023.825 s (steps_run=7)
[n=1024] Acceptance ratio: 1.000
[n=1024] Is unitary: True
[n=1024] Det magnitude: 1.000000
[n=1024] Frobenius distance: 45.267651

======================================
Running structured speed-optimized collapse for n = 2048
======================================
[n=2048] Initial block energy: 41436.251352
step   0 | current_E=40752.036051 | best_E=40752.036051 | rel_improve=1.651e-02 | step_size=8.800e-02
step   5 | current_E=39479.959173 | best_E=39479.959173 | rel_improve=1.670e-02 | step_size=1.417e-01
[n=2048] Early stopping at step 6 (rel_improve=2.406e-04)

[n=2048] Final best energy: 39470.461838
[n=2048] Total time: 3977.078 s (steps_run=7)
[n=2048] Acceptance ratio: 1.000
[n=2048] Is unitary: True
[n=2048] Det magnitude: 1.000000
[n=2048] Frobenius distance: 64.002285

======================================
Running structured speed-optimized collapse for n = 4096
======================================
[n=4096] Initial block energy: 80173.304992
step   0 | current_E=78720.748670 | best_E=78720.748670 | rel_improve=1.812e-02 | step_size=8.800e-02
step   5 | current_E=78443.923513 | best_E=78443.923513 | rel_improve=9.864e-06 | step_size=9.019e-02
[n=4096] Early stopping at step 6 (rel_improve=0.000e+00)

[n=4096] Final best energy: 78443.923513
[n=4096] Total time: 15998.157 s (steps_run=7)
[n=4096] Acceptance ratio: 0.714
[n=4096] Is unitary: True
[n=4096] Det magnitude: 1.000000
[n=4096] Frobenius distance: 90.519847
"""


import cupy as cp
import numpy as np
import time
import csv

# =========================
# Random unitary (GPU)
# =========================

def random_unitary(n):
    X = cp.random.randn(n, n) + 1j * cp.random.randn(n, n)
    Q, _ = cp.linalg.qr(X)
    return Q

# =========================
# Multiscale blocks
# =========================

def multiscale_blocks(n):
    blocks = []
    k = int(cp.log2(n))
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

def multiscale_block_sums(U, blocks):
    sums = []
    for (r0, r1, c0, c1, size) in blocks:
        B = U[r0:r1, c0:c1]
        s = cp.sum(B)
        sums.append(s)
    return cp.array(sums, dtype=complex)

# =========================
# Block energy (with precomputed s_ref)
# =========================

def block_energy(U, s_ref, blocks):
    s_U = multiscale_block_sums(U, blocks)
    diff = s_U - s_ref
    return cp.sum(cp.abs(diff)**2)

# =========================
# Structured masks for blocks
# =========================

def build_structured_masks(max_block_size=8):
    """
    Simple structured masks:
    - identity
    - all-ones
    - checkerboard (+1/-1)
    - Hadamard-like (for small sizes)
    """
    masks = {}
    for size in [2, 4, 8]:
        if size > max_block_size:
            continue
        m_list = []

        I = cp.eye(size)
        m_list.append(I)

        ones = cp.ones((size, size))
        m_list.append(ones)

        cb = cp.zeros((size, size))
        for i in range(size):
            for j in range(size):
                cb[i, j] = 1 if (i + j) % 2 == 0 else -1
        m_list.append(cb)

        if size == 2:
            H = cp.array([[1, 1],
                          [1, -1]], dtype=float)
            m_list.append(H)

        masks[size] = m_list

    return masks

# =========================
# Batch candidate descent (structured + adaptive)
# =========================

def batch_descent_step(U, s_ref, blocks, masks, step_size=0.1, num_candidates=16):
    E_current = block_energy(U, s_ref, blocks)
    best_U = U
    best_E = E_current
    accepted = False

    for _ in range(num_candidates):
        U_candidate = U.copy()

        idx = int(cp.random.randint(0, len(blocks)))
        (r0, r1, c0, c1, size) = blocks[idx]

        # pick a structured mask if available, else random
        if size in masks:
            m_list = masks[size]
            midx = int(cp.random.randint(0, len(m_list)))
            base_mask = m_list[midx]
            scale = step_size * (cp.random.randn() + 1j * cp.random.randn())
            delta_block = scale * base_mask
        else:
            delta_block = step_size * (cp.random.randn(size, size) + 1j * cp.random.randn(size, size))

        U_candidate[r0:r1, c0:c1] += delta_block

        Q, _ = cp.linalg.qr(U_candidate)
        E_candidate = block_energy(Q, s_ref, blocks)

        if E_candidate < best_E:
            best_E = E_candidate
            best_U = Q
            accepted = True

    return best_U, best_E, accepted

# =========================
# Collapse flow with metrics + adaptive step + early stopping
# =========================

def run_structured_collapse(
    n=32,
    steps=40,
    step_size_init=0.08,
    num_candidates=16,
    tol=1e-3,
    step_decay=0.7,
    step_growth=1.1
):
    U_ref = random_unitary(n)
    U = random_unitary(n)

    blocks = multiscale_blocks(n)
    s_ref = multiscale_block_sums(U_ref, blocks)
    masks = build_structured_masks(max_block_size=8)

    E0 = block_energy(U, s_ref, blocks)
    print(f"[n={n}] Initial block energy: {float(E0):.6f}")

    best_E = E0
    best_U = U.copy()

    energies = []
    times = []
    accepts = 0

    step_size = step_size_init
    start_total = time.time()

    for t in range(steps):
        iter_start = time.time()
        U, E_current, accepted = batch_descent_step(
            U, s_ref, blocks, masks, step_size=step_size, num_candidates=num_candidates
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
                f"step {t:3d} | current_E={float(E_current):.6f} | "
                f"best_E={float(best_E):.6f} | rel_improve={rel_improve:.3e} | step_size={step_size:.3e}"
            )

        if rel_improve < tol and t > 5:
            print(f"[n={n}] Early stopping at step {t} (rel_improve={rel_improve:.3e})")
            break

    end_total = time.time()
    total_time = end_total - start_total
    steps_run = len(energies)
    accept_ratio = accepts / steps_run if steps_run > 0 else 0.0

    UUdag = best_U @ best_U.conj().T
    is_unitary = cp.allclose(UUdag, cp.eye(n), atol=1e-8)

    eigvals = np.linalg.eigvals(cp.asnumpy(best_U))
    detU = np.linalg.det(cp.asnumpy(best_U))
    frob_dist = np.linalg.norm(cp.asnumpy(best_U - U_ref))

    print(f"\n[n={n}] Final best energy: {float(best_E):.6f}")
    print(f"[n={n}] Total time: {total_time:.3f} s (steps_run={steps_run})")
    print(f"[n={n}] Acceptance ratio: {accept_ratio:.3f}")
    print(f"[n={n}] Is unitary: {bool(is_unitary)}")
    print(f"[n={n}] Det magnitude: {abs(detU):.6f}")
    print(f"[n={n}] Frobenius distance: {float(frob_dist):.6f}")
    # Save final collapsed matrix for external inspection
    np.save(f"collapsed_matrix_n{n}.npy", cp.asnumpy(best_U))
    np.save(f"collapsed_matrix_eigvals_n{n}.npy", eigvals)


    metrics = {
        "n": n,
        "steps_configured": steps,
        "steps_run": steps_run,
        "num_candidates": num_candidates,
        "initial_energy": float(E0),
        "final_energy": float(best_E),
        "total_time": total_time,
        "avg_iter_time": sum(times) / len(times),
        "accept_ratio": accept_ratio,
        "is_unitary": bool(is_unitary),
        "det_mag": float(abs(detU)),
        "frob_dist": float(frob_dist),
    }

    return metrics

# =========================
# Scaling benchmark + CSV
# =========================

def scaling_benchmark(output_csv="scaling_results_structured_speed.csv"):
    #sizes = [32, 64, 128, 256, 1024]
    sizes = [2048]
    steps = 40
    step_size_init = 0.08
    num_candidates = 16

    fieldnames = [
        "n", "steps_configured", "steps_run", "num_candidates",
        "initial_energy", "final_energy",
        "total_time", "avg_iter_time",
        "accept_ratio", "is_unitary",
        "det_mag", "frob_dist"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n in sizes:
            print("\n======================================")
            print(f"Running structured speed-optimized collapse for n = {n}")
            print("======================================")

            metrics = run_structured_collapse(
                n=n,
                steps=steps,
                step_size_init=step_size_init,
                num_candidates=num_candidates
            )

            writer.writerow(metrics)

    print(f"\nStructured speed-optimized scaling results written to {output_csv}")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    cp.random.seed(0)
    scaling_benchmark()
