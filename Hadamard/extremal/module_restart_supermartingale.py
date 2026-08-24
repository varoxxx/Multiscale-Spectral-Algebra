"""
David Mulnix copyright 2026
"""

# module_restart_supermartingale.py

import numpy as np

def restart_supermartingale(
    A_history,
    F_history,
    T2_history=None,
    Spread_history=None,
):
    """
    Module 11: Restart Supermartingale (Extremal Version)

    Implements the restart rule:

        A_{k+1} = A_j  for some j < k with F(A_j) < F(A_k)

    This enforces the extremal-invariant supermartingale:

        E[F(A_{k+1}) | A_k] ≤ F(A_k)

    Parameters
    ----------
    A_history : list of np.ndarray
        List of matrices [A_0, A_1, ..., A_k].
    F_history : list of float
        List of extremal invariant values [F(A_0), F(A_1), ..., F(A_k)].
    T2_history : list of float, optional
        Quadratic trace values (for refinement only).
    Spread_history : list of float, optional
        Spectral spread values (for refinement only).

    Returns
    -------
    A_next : np.ndarray
        Restarted matrix A_{k+1}.
    j : int
        Index j < k used for restart.
    """

    k = len(A_history) - 1
    Fk = F_history[k]

    # === CORE EXTREMAL RULE (must remain exactly as written) ===
    candidates = [j for j in range(k) if F_history[j] < Fk]

    if len(candidates) == 0:
        # No restart possible; return current matrix
        return A_history[k], k

    # === If no refinement invariants provided, use original behavior ===
    if T2_history is None or Spread_history is None:
        j = candidates[0]  # earliest valid restart
        return A_history[j], j

    # === Refinement: choose best geometry among valid F-based candidates ===
    T2_candidates = np.array([T2_history[j] for j in candidates])
    Spread_candidates = np.array([Spread_history[j] for j in candidates])

    # Normalize to avoid scale issues
    T2_norm = T2_candidates / (np.max(T2_candidates) + 1e-12)
    Spread_norm = Spread_candidates / (np.max(Spread_candidates) + 1e-12)

    # Lower T2 and lower Spread preferred
    refinement_score = T2_norm + Spread_norm

    best_idx = int(np.argmin(refinement_score))
    j = candidates[best_idx]

    return A_history[j], j
