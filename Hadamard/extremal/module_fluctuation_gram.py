# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

import numpy as np
from numpy.linalg import eigh

def fluctuation_gram(A: np.ndarray):
    """
    Module 1: Fluctuation Gram and Eigenstructure

    A(A) = A A^T - n I_n

    Parameters
    ----------
    A : np.ndarray
        {±1}-matrix of shape (n, n).

    Returns
    -------
    Delta : np.ndarray
        Fluctuation Gram matrix Δ(A).
    lambdas : np.ndarray
        Eigenvalues λ_i of Δ(A).
    vecs : np.ndarray
        Eigenvectors v_i of Δ(A), columns are v_i.
    """
    n = A.shape[0]
    Delta = A @ A.T - n * np.eye(n)
    # Δ(A) is symmetric, so we use eigh
    lambdas, vecs = eigh(Delta)
    return Delta, lambdas, vecs
