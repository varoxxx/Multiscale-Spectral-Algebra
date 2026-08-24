# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# module_extremal_invariant.py

import numpy as np

def spectral_extremal_invariant(lambdas: np.ndarray):
    """
    Module 3: Spectral Extremal Invariant

    Implements:
        F(A) = M4(A) / T2(A)^2
        M4(A) = sum_i λ_i^4
        T2(A) = sum_i λ_i^2

    Parameters
    ----------
    lambdas : np.ndarray
        Eigenvalues λ_i of Δ(A), shape (n,).

    Returns
    -------
    F : float
        Spectral extremal invariant F(A).
    M4 : float
        Fourth spectral moment M4(A).
    T2 : float
        Quadratic trace invariant T2(A).
    """

    # Quadratic trace invariant T2(A)
    T2 = np.sum(lambdas**2)

    # Fourth spectral moment M4(A)
    M4 = np.sum(lambdas**4)

    # Extremal invariant F(A)
    # F(A) = M4(A) / T2(A)^2
    # Assumes T2 > 0 for non-Hadamard matrices
    if T2 == 0:
        # Exact Hadamard case: Δ(A) = 0, all λ_i = 0
        F = 0.0
    else:
        F = M4 / (T2**2)

    return float(F), float(M4), float(T2)
