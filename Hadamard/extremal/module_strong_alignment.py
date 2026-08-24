# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# module_strong_alignment.py

import numpy as np

def strong_spectral_alignment(A: np.ndarray, P_ext: np.ndarray):
    """
    Module 8: Strong Spectral Alignment (Extremal Version)

    Implements:
        A_strong = sign(P_ext * A * P_ext^T)

    where the sign map is applied entrywise:
        sign(x) = +1 if x >= 0
        sign(x) = -1 if x < 0

    Zeros are mapped to +1.

    Parameters
    ----------
    A : np.ndarray
        {±1}-matrix of shape (n, n).
    P_ext : np.ndarray
        Blended extremal projector P_ext of shape (n, n).

    Returns
    -------
    A_strong : np.ndarray
        Strongly aligned {±1}-matrix.
    """

    # Bilinear projection onto dominant row and column extremal modes
    PAP = P_ext @ A @ P_ext.T

    # Entrywise sign map, zeros → +1
    A_strong = np.where(PAP >= 0, 1, -1)

    return A_strong
