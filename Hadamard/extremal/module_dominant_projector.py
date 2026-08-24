# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:33:11 2026

@author: Varoc
"""

# module_dominant_projector.py
"""
David Mulnix copyright 2026
"""
import numpy as np

def dominant_extremal_projector(lambdas: np.ndarray, vecs: np.ndarray, sigma: float = 0.8):
    """
    Module 5: Dominant Spectral Projector (Extremal Version)

    Constructs the blended extremal projector:

        J_ext  = { i : largest extremal ratios λ_i^4 / T2^2 }
        P_ext  = sum_{i in J_ext} w_i v_i v_i^T
        sum_{i in J_ext} w_i = 1

    Parameters
    ----------
    lambdas : np.ndarray
        Eigenvalues λ_i of Δ(A), shape (n,).
    vecs : np.ndarray
        Eigenvectors v_i of Δ(A), shape (n, n). Columns are v_i.
    sigma : float, optional
        Fraction of modes to keep in the dominant extremal set J_ext,
        typically in [0.75, 0.85]. Default is 0.8.

    Returns
    -------
    P_ext : np.ndarray
        Blended extremal projector P_ext of shape (n, n).
    J_ext : np.ndarray
        Indices of dominant extremal modes.
    w : np.ndarray
        Weights w_i used in the projector.
    """

    n = lambdas.shape[0]

    # Compute T2(A)
    T2 = np.sum(lambdas**2)

    # Extremal ratios r_i = λ_i^4 / T2^2
    if T2 == 0:
        # Hadamard case: no dominant modes, projector is zero
        return np.zeros((n, n)), np.array([], dtype=int), np.array([])

    ratios = lambdas**4 / (T2**2)

    # Select dominant indices J_ext by largest extremal ratios
    k = max(1, int(np.round(sigma * n)))
    J_ext = np.argsort(ratios)[-k:]  # largest k ratios

    # Normalize weights over J_ext
    r_J = ratios[J_ext]
    if np.all(r_J == 0):
        # Degenerate case: fall back to uniform weights
        w = np.ones_like(r_J, dtype=float) / r_J.size
    else:
        w = r_J / np.sum(r_J)

    # Build P_ext = sum_{i in J_ext} w_i v_i v_i^T
    P_ext = np.zeros((n, n))
    for idx, wi in zip(J_ext, w):
        v = vecs[:, idx].reshape(-1, 1)
        P_ext += wi * (v @ v.T)

    return P_ext, J_ext, w
