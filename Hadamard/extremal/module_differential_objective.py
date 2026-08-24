# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:35:37 2026

@author: Varoc
"""

"""
David Mulnix copyright 2026
"""


# module_differential_objective.py

import numpy as np

def differential_collapse_objective(
    Delta: np.ndarray,
    lambdas: np.ndarray,
    vecs: np.ndarray,
    J_ext: np.ndarray,
    F: float,
):
    """
    Module 9: Differential Collapse Objective (Extremal Version)

    Implements:
        D_spec = sum_{i in J_ext} (v_i^T Δ(A) v_i)^2
        D_inv  = F(A)^2 + ||Δ(A)||_F^2 + Spread(A)^2
        D_mode = sum_{i=1}^n (v_i^T Δ(A) v_i)^2

    Parameters
    ----------
    Delta : np.ndarray
        Fluctuation Gram matrix Δ(A), shape (n, n).
    lambdas : np.ndarray
        Eigenvalues λ_i of Δ(A), shape (n,).
    vecs : np.ndarray
        Eigenvectors v_i of Δ(A), shape (n, n). Columns are v_i.
    J_ext : np.ndarray
        Indices of dominant extremal modes.
    F : float
        Spectral extremal invariant F(A).

    Returns
    -------
    objectives : dict
        {
            "D_spec": float,
            "D_inv": float,
            "D_mode": float
        }
    """

    n = Delta.shape[0]

    # Spread(A) = λ_max - λ_min
    lam_max = np.max(lambdas)
    lam_min = np.min(lambdas)
    Spread = lam_max - lam_min

    # Frobenius norm ||Δ(A)||_F
    norm_Delta_sq = np.linalg.norm(Delta, ord='fro')**2

    # D_spec = sum_{i in J_ext} (v_i^T Δ v_i)^2
    D_spec = 0.0
    for i in J_ext:
        v = vecs[:, i]
        val = float(v.T @ (Delta @ v))
        D_spec += val**2

    # D_mode = sum_{i=1}^n (v_i^T Δ v_i)^2
    D_mode = 0.0
    for i in range(n):
        v = vecs[:, i]
        val = float(v.T @ (Delta @ v))
        D_mode += val**2

    # D_inv = F(A)^2 + ||Δ(A)||_F^2 + Spread(A)^2
    D_inv = F**2 + norm_Delta_sq + Spread**2

    return {
        "D_spec": float(D_spec),
        "D_inv": float(D_inv),
        "D_mode": float(D_mode),
    }
