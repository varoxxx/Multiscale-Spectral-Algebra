"""
David Mulnix copyright 2026
"""

# module_global_invariants.py

import numpy as np

def global_invariants(Delta: np.ndarray, lambdas: np.ndarray, offdiag_values: np.ndarray = None):
    """
    Module 2: Global Invariants (Safe Version)

    Implements:
        T2(A)      = sum_i λ_i^2
        Spread(A)  = λ_max - λ_min
        H(A)       = - sum_v p(v) log p(v)
        Rdet(A)    = exp( (1/n) * sum_i log(1 + λ_i) )   [SAFE]
        L(A)       = - log Rdet(A)
        dHad(A)    = sqrt(T2(A))

    This version is robust against:
        • log(1 + λ_i) <= 0
        • NaN propagation
        • empty off-diagonal sets
        • invalid determinant states

    All invariants are guaranteed finite or NaN-safe.
    """

    n = Delta.shape[0]

    # ---------------------------------------------------------
    # T2(A) = sum λ_i^2
    # ---------------------------------------------------------
    T2 = np.sum(lambdas**2)

    # ---------------------------------------------------------
    # Spread(A) = λ_max - λ_min
    # ---------------------------------------------------------
    lam_max = np.max(lambdas)
    lam_min = np.min(lambdas)
    Spread = lam_max - lam_min

    # ---------------------------------------------------------
    # Tier entropy H(A)
    # ---------------------------------------------------------
    if offdiag_values is None:
        mask = ~np.eye(n, dtype=bool)
        offdiag_values = Delta[mask]

    if offdiag_values.size == 0:
        H = 0.0
    else:
        vals, counts = np.unique(offdiag_values, return_counts=True)
        p = counts.astype(float) / offdiag_values.size

        # Safe entropy: avoid log(0)
        p_safe = np.where(p > 0, p, 1.0)
        H = -np.sum(p * np.log(p_safe))

    # ---------------------------------------------------------
    # SAFE determinant Rdet(A)
    #
    # PDF assumption:
    #   “We assume (1 + λ_i) > 0 for admissible states.”
    #
    # Random matrices violate this, so we clamp safely.
    # ---------------------------------------------------------
    one_plus = 1.0 + lambdas
    valid = one_plus > 0

    if np.any(valid):
        log_Rdet = (1.0 / n) * np.sum(np.log(one_plus[valid]))
        Rdet = np.exp(log_Rdet)
        L = -log_Rdet
    else:
        # No valid determinant state
        log_Rdet = float('nan')
        Rdet = float('nan')
        L = float('nan')

    # ---------------------------------------------------------
    # Hadamard distance dHad(A) = sqrt(T2)
    # ---------------------------------------------------------
    dHad = np.sqrt(T2)

    return {
        "T2": float(T2),
        "Spread": float(Spread),
        "H": float(H),
        "Rdet": float(Rdet),
        "L": float(L),
        "dHad": float(dHad),
    }
