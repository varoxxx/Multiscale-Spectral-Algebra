# -*- coding: utf-8 -*-
"""
David Mulnix Copyright 2026
"""

import numpy as np
import csv


# ============================================================
# Core utilities
# ============================================================

"""
gram_deviation(A) computes the fluctuation Gram matrix:

    Δ(A) = A Aᵀ − nI

In the math, Δ(A) is the *defect operator* measuring deviation
from Hadamard structure. A true Hadamard satisfies Δ(A)=0.

This operator is the foundation of the collapse landscape:
Movement A attempts flips that reduce ||Δ(A)||_F.
"""
def gram_deviation(A):
    n = A.shape[0]
    return A @ A.T - n * np.eye(n)

"""
D(A) = ||Δ(A)||_F is the collapse invariant.

In the math, D(A) is the scalar that tracks progress through
the collapse landscape. When D(A)=0, the matrix is Hadamard.

Movement A is a greedy descent on D(A).
"""
def D(A):
    return np.linalg.norm(gram_deviation(A), "fro")

"""
sylvester_hadamard(k) constructs H_{2^k}.

These lower-order Hadamards (H4, H8) serve as *spectral
fingerprints* in your framework. They are not templates to
match, but invariant spectral anchors used to detect alignment
inside higher-dimensional spectral neighborhoods.
"""
def sylvester_hadamard(k):
    H = np.array([[1, 1],
                  [1, -1]], dtype=int)
    for _ in range(1, k):
        H = np.block([[H, H],
                      [H, -H]])
    return H

H4 = sylvester_hadamard(2)
H8 = sylvester_hadamard(3)

# ============================================================
# Spectral space (n8) + SA1 anchors (n4 + n8)
# ============================================================

"""
spectral_space(A) computes the eigenvalues/eigenvectors of Δ(A).

Mathematically:
- eigenvalues correspond to defect magnitudes
- eigenvectors encode *spectral mass distribution*
- sorting by |λ| identifies dominant defect modes

These dominant modes define the spectral-density block.
"""
def spectral_space(A, n_eig=8):
    Delta = gram_deviation(A)
    eigvals, eigvecs = np.linalg.eigh(Delta)
    idx = np.argsort(np.abs(eigvals))
    return eigvals[idx], eigvecs[:, idx]

"""
spectral_density_block(A) extracts the block where spectral mass
is concentrated.

This is the “defect core” in your math:
- rows/cols chosen by eigenvector magnitude
- block is the reduced search space
- collapse acts only inside this block

This implements your concept of *spectral-density blocks* and
reduces the effective search space to meaningful directions.
"""
def spectral_density_block(A, block_size=12, n_eig=8):
    eigvals, eigvecs = spectral_space(A, n_eig=n_eig)
    V = eigvecs[:, -n_eig:]
    scores = np.sum(np.abs(V), axis=1)
    idx = np.argsort(-scores)
    rows = np.sort(idx[:block_size])
    cols = np.sort(idx[:block_size])
    return rows, cols, scores, eigvals, eigvecs

"""
corr_with_H4 measures alignment between the spectral block and
the H4 fingerprint.

This is your *lower-order spectral fingerprint* mechanism:
detecting whether local spectral geometry resembles H4.
"""
def corr_with_H4(A, rows, cols):
    if len(rows) < 4 or len(cols) < 4:
        return None
    return np.sum(A[np.ix_(rows[:4], cols[:4])] * H4)

"""
corr_with_H8 measures alignment with the H8 fingerprint.

This is the second anchor in your SA1 system. Together H4/H8
provide multiscale spectral guidance for collapse.
"""
def corr_with_H8(A, rows, cols):
    if len(rows) < 8 or len(cols) < 8:
        return None
    return np.sum(A[np.ix_(rows[:8], cols[:8])] * H8)

# ============================================================
# Movement A greedy step
# ============================================================

"""
greedy_step_A implements Movement A:

- iterate over entries in the spectral-density block
- flip A[i,j] → -A[i,j]
- accept flip only if it reduces D(A)

This is your *local spectral descent* operator:
a discrete analogue of gradient descent inside the reduced
spectral neighborhood.

Movement A is guided by:
- spectral-density block (defect core)
- lower-order fingerprints (alignment)
"""
def greedy_step_A(A, rows, cols):
    best_A = A
    best_D = D(A)
    accepted = 0
    attempted = 0

    for i in rows:
        for j in cols:
            attempted += 1
            B = A.copy()
            B[i, j] *= -1
            DB = D(B)
            if DB < best_D:
                best_D = DB
                best_A = B
                accepted += 1

    return best_A, attempted, accepted

# ============================================================
# Diagnostics writer
# ============================================================

diag_file = open("movementA_diagnostics_extended.csv", "w", newline="", encoding="utf-8")
diag_writer = csv.DictWriter(diag_file, fieldnames=[
    "trial","movement","block_size","realign_freq","seed",
    "event_type","step","D",
    "block_energy","block_energy_fraction",
    "delta_D","corr_H4","corr_H8",
    "rows","cols","overlap",
    "eigvals","eigvec_norms",
    "attempted_flips","accepted_flips",
    "q11","q12","q21","q22",
    "diag_energy","offdiag_energy"
])
diag_writer.writeheader()

# ============================================================
# Enhanced diagnostics (logs every step)
# ============================================================

"""
diagnostics_every_step logs all spectral quantities relevant to
your mathematical framework:

- D(A): collapse invariant
- block_energy: energy inside defect core
- quadrant energies: internal block geometry
- diag/offdiag energies: structure of Δ(A)
- correlations with H4/H8: fingerprint alignment
- eigenvalues/eigenvector norms: spectral mass transport
- overlap: realignment stability

This produces a full trace of the collapse trajectory.
"""
def diagnostics_every_step(trial, A, rows, cols, movement, k, rf, seed,
                           event_type, step, attempted=None, accepted=None,
                           overlap=None):

    Delta = gram_deviation(A)

    block_energy = np.linalg.norm(Delta[np.ix_(rows, cols)], "fro")
    total_energy = np.linalg.norm(Delta, "fro")
    block_fraction = block_energy / total_energy if total_energy > 0 else 0.0

    qsize = len(rows) // 2
    if qsize > 0:
        q11 = np.linalg.norm(Delta[np.ix_(rows[:qsize], cols[:qsize])], "fro")
        q12 = np.linalg.norm(Delta[np.ix_(rows[:qsize], cols[qsize:])], "fro")
        q21 = np.linalg.norm(Delta[np.ix_(rows[qsize:], cols[:qsize])], "fro")
        q22 = np.linalg.norm(Delta[np.ix_(rows[qsize:], cols[qsize:])], "fro")
    else:
        q11 = q12 = q21 = q22 = None

    diag_energy = np.linalg.norm(np.diag(np.diag(Delta)), "fro")
    offdiag_energy = np.linalg.norm(Delta - np.diag(np.diag(Delta)), "fro")

    c4 = corr_with_H4(A, rows, cols)
    c8 = corr_with_H8(A, rows, cols)

    eigvals, eigvecs = spectral_space(A, n_eig=8)
    eigvec_norms = [float(np.linalg.norm(v)) for v in eigvecs.T]

    diag_writer.writerow({
        "trial": trial,
        "movement": movement,
        "block_size": k,
        "realign_freq": rf,
        "seed": seed,
        "event_type": event_type,
        "step": step,
        "D": D(A),
        "block_energy": block_energy,
        "block_energy_fraction": block_fraction,
        "delta_D": None,
        "corr_H4": c4,
        "corr_H8": c8,
        "rows": ",".join(map(str, rows)),
        "cols": ",".join(map(str, cols)),
        "overlap": overlap,
        "eigvals": eigvals.tolist(),
        "eigvec_norms": eigvec_norms,
        "attempted_flips": attempted,
        "accepted_flips": accepted,
        "q11": q11,
        "q12": q12,
        "q21": q21,
        "q22": q22,
        "diag_energy": diag_energy,
        "offdiag_energy": offdiag_energy
    })

# ============================================================
# Movement A runner (patched to log every step)
# ============================================================

"""
run_single_trial executes the full collapse process:

- initialize A
- extract spectral-density block
- seed block with noise or H8 (SA1 anchor)
- run Movement A for many steps
- periodically realign block (spectral mass transport)
- log everything
- save Hadamards when D(A)=0

This function implements your entire collapse pipeline:
spectral fingerprints → block extraction → Movement A →
realignment → collapse corridor → Hadamard.
"""
def run_single_trial(trial, n=12, steps=600):
    block_sizes = [8, 12]
    realign_freq = 10
    seeds = ["random", "sylvester8"]

    results = []

    for k in block_sizes:
        for seed in seeds:

            A = np.random.choice([-1,1], size=(n,n))
            rows, cols, scores, eigvals, eigvecs = spectral_density_block(A, block_size=k, n_eig=8)

            if seed == "random":
                A[np.ix_(rows, cols)] = np.random.choice([-1,1], size=(k,k))
            elif seed == "sylvester8" and k >= 8:
                A[np.ix_(rows[:8], cols[:8])] = H8

            diagnostics_every_step(trial, A, rows, cols, "A", k, realign_freq, seed,
                                   "initial", 0)

            for t in range(1, steps+1):

                A, attempted, accepted = greedy_step_A(A, rows, cols)

                diagnostics_every_step(trial, A, rows, cols, "A", k, realign_freq, seed,
                                       "step", t, attempted, accepted)

                if t % realign_freq == 0:
                    rows2, cols2, scores2, eigvals2, eigvecs2 = spectral_density_block(A, block_size=k, n_eig=8)
                    overlap = len(set(rows) & set(rows2))

                    diagnostics_every_step(trial, A, rows, cols, "A", k, realign_freq, seed,
                                           "realign_before", t, attempted, accepted,
                                           overlap=overlap)

                    rows, cols = rows2, cols2

                    diagnostics_every_step(trial, A, rows, cols, "A", k, realign_freq, seed,
                                           "realign_after", t, attempted, accepted,
                                           overlap=overlap)

            final_D = D(A)
            results.append((trial, "A", k, realign_freq, seed, final_D))

            if final_D == 0:
                fname = f"hadamard_trial_{trial}_block_{k}_seed_{seed}.csv"
                np.savetxt(fname, A, fmt="%d", delimiter=",")
                print(f"Saved Hadamard matrix: {fname}")

    return results

# ============================================================
# Multi-trial runner
# ============================================================

"""
main() runs multiple collapse trials and writes summary files.

This produces:
- movementA_summary_extended.csv
- movementA_diagnostics_extended.csv

Together these files contain the full spectral trace of collapse,
matching the mathematical narrative in your paper.
"""
def main():
    np.random.seed(0)

    N_trials = 10
    all_results = []

    for trial in range(1, N_trials+1):
        print(f"Running trial {trial}")
        res = run_single_trial(trial)
        all_results.extend(res)

    with open("movementA_summary_extended.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial","movement","block_size","realign_freq","seed","final_D"])
        for r in all_results:
            w.writerow(r)

    print("movementA_summary_extended.csv written")
    print("movementA_diagnostics_extended.csv written")

if __name__ == "__main__":
    main()
    diag_file.close()
