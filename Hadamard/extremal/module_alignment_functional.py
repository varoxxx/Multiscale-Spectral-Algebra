# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# module_alignment_functional.py

import numpy as np

def extremal_alignment_functional(Delta: np.ndarray, P_ext: np.ndarray):
    """
    Module 6: Alignment Functional (Extremal Version)

    Implements:
        a_ext(A) = <Δ(A), P_ext>_F / ( ||Δ(A)||_F * ||P_ext||_F )

    Parameters
    ----------
    Delta : np.ndarray
        Fluctuation Gram matrix Δ(A), shape (n, n).
    P_ext : np.ndarray
        Blended extremal projector P_ext, shape (n, n).

    Returns
    -------
    a_ext : float
        Extremal alignment functional a_ext(A).
    """

    # Frobenius inner product <Δ, P_ext>_F
    num = np.sum(Delta * P_ext)

    # Frobenius norms
    norm_Delta = np.linalg.norm(Delta, ord='fro')
    norm_P = np.linalg.norm(P_ext, ord='fro')

    if norm_Delta == 0 or norm_P == 0:
        return 0.0

    a_ext = num / (norm_Delta * norm_P)
    return float(a_ext)
