"""
David Mulnix Copyright 2026
"""
import numpy as np

# Collapse operator on Gram defect:
# Iteratively removes contribution from extremal eigenvectors (contraction law).
def collapse_gram(Delta, k_ext=2, verbose=False):
    lamb, vecs = np.linalg.eigh(Delta)


    idx_sorted = np.argsort(-np.abs(lamb))
    idx_ext = idx_sorted[:k_ext]


    P_ext = np.zeros_like(Delta)
    for i in idx_ext:
        v = vecs[:, i].reshape(-1, 1)
        P_ext += v @ v.T


    Delta_ext = P_ext @ Delta @ P_ext
    Delta_new = Delta - Delta_ext


    if verbose:
        lamb_new = np.linalg.eigvalsh(Delta_new)
        print("Extremal eigenvalues removed:", lamb[idx_ext])
        print("T2_old =", np.sum(lamb**2))
        print("T2_new =", np.sum(lamb_new**2))
        print("Reduction factor =", np.sum(lamb_new**2)/np.sum(lamb**2))
        print("||Delta_new||_F =", np.linalg.norm(Delta_new))


    return Delta_new   # *** ONLY THIS ***

# ============================================================
# 2. Build multiscale block hierarchy for Sylvester H_n
# ============================================================
# Multiscale Sylvester block hierarchy:
# Defines dyadic blocks 2,4,8,... used for backbone and sign algebra.

def multiscale_blocks(n):
    """
    Returns list of (row_start, row_end, col_start, col_end, block_size)
    for all Sylvester blocks at scales 2,4,8,...,n.
    """
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




# ============================================================
# 3. Multiscale Sign Algebra: build D_r and D_c from m(n)
# ============================================================

def build_Dr_Dc(n, m):
    """
    Build diagonal sign matrices D_r and D_c from multiscale sign algebra.
    m is the full multiscale sign template of length M_n.
    """
    blocks = multiscale_blocks(n)
    M = len(blocks)
    if len(m) != M:
        raise ValueError(f"Motif template length {len(m)} does not match M_n={M}")


    # Row/column sign accumulators
    row_signs = np.ones(n, dtype=int)
    col_signs = np.ones(n, dtype=int)


    for idx, (rs, re, cs, ce, size) in enumerate(blocks):
        sign = m[idx]
        # Apply sign to all rows in this block
        row_signs[rs:re] *= sign
        # Apply sign to all columns in this block
        col_signs[cs:ce] *= sign


    Dr = np.diag(row_signs)
    Dc = np.diag(col_signs)
    return Dr, Dc

# Reconstruct matrix from Gram:
# Standard spectral reconstruction, used after Gram collapse.
def reconstruct_from_gram(Delta, n):
    G = Delta + n * np.eye(n)
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    Sigma_sqrt = np.diag(np.sqrt(eigvals_clipped))
    A_rec = eigvecs @ Sigma_sqrt @ eigvecs.T
    return A_rec


# Multi-jump Gram collapse:
# Applies collapse_gram repeatedly to contract defect, then reconstructs A_rec.
def multi_jump_collapse(A0, n, num_jumps=8, k_ext=2, verbose=False):
    Delta = A0 @ A0.T - n * np.eye(n)

    for j in range(num_jumps):
        Delta = collapse_gram(Delta, k_ext=k_ext, verbose=verbose)

    A_rec = reconstruct_from_gram(Delta, n)

    return A_rec, Delta

# ============================================================
# Deterministic multiscale template m(A_rec)
# ============================================================
# For each block, take sum of entries and assign sign; this defines m(A_rec).
def deterministic_multiscale_template(A_rec, n):
    """
    Deterministic multiscale sign template m(A_rec).


    For each multiscale block B in A_rec, we compute:
        s_B = sum of entries in B
        m_B = sign(s_B)  (with tie-breaking to +1)


    This yields a deterministic m of length M_n.
    """
    blocks = multiscale_blocks(n)
    m = np.zeros(len(blocks), dtype=int)


    for idx, (r0, r1, c0, c1, size) in enumerate(blocks):
        B = A_rec[r0:r1, c0:c1]
        s = np.sum(B)
        if s > 0:
            m[idx] = 1
        elif s < 0:
            m[idx] = -1
        else:
            m[idx] = 1  # tie-break to +1


    return m

# Standard Sylvester Hadamard (used only for dyadic blocks in backbone).
def sylvester_hadamard(n):
    """Generate Sylvester Hadamard matrix of order n (n must be power of 2)."""
    if n == 1:
        return np.array([[1]], dtype=int)
    H = sylvester_hadamard(n // 2)
    return np.block([[H, H], [H, -H]])

# Pure closed-form backbone Q_n:
# Sylvester tiling + invariant-based patch using row/col sums of A_rec.
def build_block_closed_form_pure_backbone(A_rec, n):
    """
    Pure closed-form backbone:
    - Sylvester blocks for dyadic sizes
    - leftover rows/cols filled by invariant signs (row_sums, col_sums)
    - no multiscale, no corridor, no Gram correction
    """
    # 1. Sylvester tiling
    sizes = []
    remaining = n
    while remaining >= 2:
        possible = [2, 4, 8, 16, 32, 64, 128]
        possible = [s for s in possible if s <= remaining]
        if not possible:
            break
        s = max(possible)
        sizes.append(s)
        remaining -= s


    blocks = [sylvester_hadamard(s) for s in sizes]
    B = np.zeros((n, n), dtype=int)


    r = c = 0
    for H in blocks:
        s = H.shape[0]
        B[r:r+s, c:c+s] = H
        r += s
        c += s


    block_rows_end = r
    block_cols_end = c


    # 2. Invariant-based patch (row/col sums)
    row_sums = np.sum(A_rec, axis=1)
    col_sums = np.sum(A_rec, axis=0)


    p = np.sign(row_sums)
    p[p == 0] = 1


    q = np.sign(col_sums)
    q[q == 0] = 1


    # leftover rows
    for i in range(block_rows_end, n):
        B[i, :] = p[i] * q


    # leftover cols
    for j in range(block_cols_end, n):
        B[:, j] = p * q[j]


    return B

# Apply Q_n: backbone + multiscale sign operators Dr, Dc.
def Qn_block_closed_form_pure(A_rec, n):
    """
    Apply Q_n using the pure closed-form backbone.
    """
    Bn = build_block_closed_form_pure_backbone(A_rec, n)


    # multiscale template + diagonal sign operators
    m = deterministic_multiscale_template(A_rec, n)
    Dr, Dc = build_Dr_Dc(n, m)


    return Dr @ Bn @ Dc



# Corridor patching for n=12:
# Uses extremal eigenspace of backbone to build patch rows/cols 0..7
# while maintaining orthogonality to corridor spine 8..11.
def corridor_patch_backbone_12(A_rec):
    n = 12
    B = Qn_block_closed_form_pure(A_rec, n)


    # corridor spine: rows/cols 8..11
    spine_rows = np.arange(8, 12)
    spine_cols = np.arange(8, 12)


    # extremal eigenspace
    eigvals, eigvecs = extremal_eigenspace(B)
    v1, v2 = eigvecs[:, 0], eigvecs[:, 1]


    # build patch rows 0..7
    patch_rows = np.arange(0, 8)
    for i in patch_rows:
        # weights from curve
        w1, w2 = v1[i], v2[i]
        candidate = np.sign(w1 * v1 + w2 * v2)


        # enforce approximate orthogonality to spine rows
        for s in spine_rows:
            ip = np.dot(candidate, B[s, :])
            if ip != 0:
                candidate *= -np.sign(ip)


        B[i, :] = candidate


    # build patch cols 0..7 similarly
    patch_cols = np.arange(0, 8)
    for j in patch_cols:
        w1, w2 = v1[j], v2[j]
        candidate = np.sign(w1 * v1 + w2 * v2)


        for s in spine_cols:
            ip = np.dot(B[:, s], candidate)
            if ip != 0:
                candidate *= -np.sign(ip)


        B[:, j] = candidate


    return np.sign(B)



# Adaptive multiscale flow on corridor:
# Level-1 jumping: local 2x2 Sylvester motif updates inside corridor,
# guided by Gram error and invariant match after multi-jump collapse.
def adaptive_multiscale_flow(B_init, corridor_idx, max_iters=50, alpha=1.0, beta=1.0):
    """
    Adaptive multi-scale tensor flow on the corridor:
    - at each step, try all local 2x2 Sylvester updates
    - evaluate Gram error and invariance after collapse
    - choose the update that best improves alpha*Gram - beta*Match
    - stop when no candidate improves the objective
    """
    B = B_init.copy()
    history = []


    for t in range(max_iters):
        n = B.shape[0]
        C = corridor_idx


        # current metrics
        ge = gram_error(B)
        eigvals, eigvecs = extremal_eigenspace(B)
        B_pert = perturb_along_curve(B, eigvecs, 0.2)
        B_rec, _ = multi_jump_collapse(B_pert, n, num_jumps=8, k_ext=2, verbose=False)
        match, row_match, col_match = compare_invariants(B, B_rec)


        obj_current = alpha * ge - beta * match


        history.append({
            "iter": t,
            "gram": ge,
            "global_match": match,
            "corridor_match": np.mean(row_match[C]),
            "patch_match": np.mean(row_match[np.arange(8, 12)]),
            "eigvals": eigvals,
            "objective": obj_current
        })


        # generate corridor candidates
        C_block = B[np.ix_(C, C)]
        candidates = candidate_updates(C_block)
        if not candidates:
            break


        best_obj = obj_current
        best_C = None


        # evaluate each candidate
        for (r0, c0, C_new) in candidates:
            B_candidate = B.copy()
            B_candidate[np.ix_(C, C)] = C_new


            ge_c = gram_error(B_candidate)
            eigvals_c, eigvecs_c = extremal_eigenspace(B_candidate)
            B_pert_c = perturb_along_curve(B_candidate, eigvecs_c, 0.2)
            B_rec_c, _ = multi_jump_collapse(B_pert_c, n, num_jumps=8, k_ext=2, verbose=False)
            match_c, row_match_c, col_match_c = compare_invariants(B_candidate, B_rec_c)


            obj_c = alpha * ge_c - beta * match_c


            if obj_c < best_obj:
                best_obj = obj_c
                best_C = C_new


        # stop if no improvement
        if best_C is None:
            break


        # apply best corridor update
        B[np.ix_(C, C)] = best_C


    return B, history



# Gram error (Frobenius norm of G - nI).
def gram_error(H):
    n = H.shape[0]
    G = H @ H.T
    return np.linalg.norm(G - n * np.eye(n), "fro")

# Invariant match between H and H_rec (global, row, column).
def compare_invariants(H, H_rec):
    match = np.mean(H == H_rec)
    row_match = np.mean(H == H_rec, axis=1)
    col_match = np.mean(H == H_rec, axis=0)
    return match, row_match, col_match

# Extremal eigenspace of Gram defect: used for corridor patching and perturbation.
def extremal_eigenspace(H):
    n = H.shape[0]
    Delta = H @ H.T - n * np.eye(n)
    vals, vecs = np.linalg.eigh(Delta)
    idx = np.argsort(np.abs(vals))[-2:]
    return vals[idx], vecs[:, idx]

# Perturb along eigenvector curve and re-sign: explores nearby configurations.
def perturb_along_curve(H, eigvecs, eps):
    v = eigvecs[:, 0]
    P = np.outer(v, v)
    H_pert = np.sign(H + eps * P)
    return H_pert



# Sylvester H2 motif (2x2 block).
def H2():
    return np.array([[1, 1],
                     [1, -1]], dtype=float)



# Align motif to local block via row/col sign adjustments.
def align_motif(block, motif):
    B = block.astype(float)
    M = motif.copy()
    row_signs = np.sign(np.sum(B * M, axis=1))
    col_signs = np.sign(np.sum(B * M, axis=0))
    row_signs[row_signs == 0] = 1
    col_signs[col_signs == 0] = 1
    M_aligned = (row_signs[:, None] * M) * col_signs[None, :]
    return np.sign(M_aligned)

# Generate candidate corridor updates via 2x2 Sylvester motifs.
def candidate_updates(C_block):
    H2_m = H2()
    candidates = []
    for r0 in range(0, 8, 2):
        for c0 in range(0, 8, 2):
            sub = C_block[r0:r0+2, c0:c0+2]
            aligned = align_motif(sub, H2_m)
            if not np.array_equal(aligned, sub):
                C_new = C_block.copy()
                C_new[r0:r0+2, c0:c0+2] = aligned
                candidates.append((r0, c0, C_new))
    return candidates


# Embed optimized corridor block back into full backbone.
def embed_full_matrix(C_corridor, backbone):
    B = backbone.copy()
    B[0:8, 0:8] = C_corridor
    B[B == 0] = 1
    return B

# Gram energy (squared Frobenius norm of G - nI).
def gram_energy(H):
    n = H.shape[0]
    G = H @ H.T
    return np.linalg.norm(G - n * np.eye(n), "fro")**2

# Random flip, biased to corridor rows.
def random_flip(H, corridor_idx=None, p_corridor=0.7):
    H_new = H.copy()
    n = H.shape[0]


    # choose row
    if corridor_idx is not None and np.random.rand() < p_corridor:
        r = np.random.choice(corridor_idx)
    else:
        r = np.random.randint(0, n)


    # choose column
    c = np.random.randint(0, n)


    H_new[r, c] *= -1
    return H_new

# Discrete annealing inside corridor:
# Refines H_init to lower Gram energy, finishing collapse to true ±1 Hadamard.
def anneal_discrete_hadamard(H_init, corridor_idx=None,
                             max_iters=50000,
                             T_start=2.0,
                             T_end=0.01):
    H = H_init.copy()
    E = gram_energy(H)


    for t in range(1, max_iters + 1):
        T = T_start * (T_end / T_start) ** (t / max_iters)  # exponential cooling


        H_candidate = random_flip(H, corridor_idx=corridor_idx)
        E_candidate = gram_energy(H_candidate)


        dE = E_candidate - E
        if dE < 0 or np.random.rand() < np.exp(-dE / max(T, 1e-6)):
            H, E = H_candidate, E_candidate


    return H, E


# Main pipeline:
# 1) Float collapse via multi-jump contraction + Q_n + corridor patch + flow.
# 2) Project to ±1.
# 3) Anneal inside corridor until true Hadamard is found.
def main():
    n = 12
    corridor_idx = np.arange(0, 8)


    # ------------------------------------------------------------
    # 1. Build your float Hadamard (your existing pipeline)
    # ------------------------------------------------------------
    A0 = np.random.choice([-1, 1], size=(n, n))
    A_rec, _ = multi_jump_collapse(A0, n, num_jumps=8, k_ext=2, verbose=False)
    backbone = corridor_patch_backbone_12(A_rec)


    B_opt, hist = adaptive_multiscale_flow(
        backbone,
        corridor_idx,
        max_iters=200,
        alpha=1.0,
        beta=1.0
    )


    C_corridor = B_opt[np.ix_(corridor_idx, corridor_idx)]
    full_candidate = embed_full_matrix(C_corridor, backbone)


    H_float, _ = multi_jump_collapse(full_candidate, n,
                                     num_jumps=20, k_ext=3, verbose=False)


    print("Float Hadamard:")
    print(H_float)
    print("Float Gram error:", gram_energy(H_float))


    # ------------------------------------------------------------
    # 2. Convert float Hadamard to initial ±1 matrix
    # ------------------------------------------------------------
    H_int = np.sign(H_float)
    H_int[H_int == 0] = 1


    print("\nInitial ±1 projection:")
    print(H_int)
    print("Initial discrete Gram energy:", gram_energy(H_int))


    # ------------------------------------------------------------
    # 3. Repeated annealing attempts until success
    # ------------------------------------------------------------
    max_attempts = 50
    max_iters = 80000
    T_start = 2.0
    T_end = 0.05


    print("\nStarting repeated annealing attempts...\n")


    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt}...")


        H_disc, E_disc = anneal_discrete_hadamard(
            H_int,
            corridor_idx=corridor_idx,
            max_iters=max_iters,
            T_start=T_start,
            T_end=T_end
        )


        print(f"  Final energy: {E_disc}")


        if np.allclose(H_disc @ H_disc.T, n * np.eye(n), atol=1e-8):
            print("\n====================================")
            print("SUCCESS — TRUE ±1 HADAMARD FOUND!")
            print("====================================\n")


            print("Discrete Hadamard (±1):")
            print(H_disc)


            # ------------------------------------------------------------
            # 4. Save to text file
            # ------------------------------------------------------------
            filename = "hadamard_12.txt"
            with open(filename, "w") as f:
                for row in H_disc:
                    f.write(" ".join(str(int(x)) for x in row) + "\n")


            print(f"\nSaved Hadamard to {filename}")
            print("Attempts needed:", attempt)
            return


    print("\nFAILED — no Hadamard found after all attempts.")




if __name__ == "__main__":
    main()



