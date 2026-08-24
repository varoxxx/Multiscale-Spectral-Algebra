# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# module_spectral_alignment.py

import numpy as np

def spectral_alignment(A: np.ndarray, P_ext: np.ndarray):
    """
    Module 7: Spectral Alignment Operator (Extremal Version)

    Implements:
        A_align = sign(P_ext * A)

    where the sign map is applied entrywise:
        sign(x) = +1 if x >= 0
        sign(x) = -1 if x < 0

    Zeros are mapped to +1, consistent with the closed algebra.

    Parameters
    ----------
    A : np.ndarray
        {±1}-matrix of shape (n, n).
    P_ext : np.ndarray
        Blended extremal projector P_ext of shape (n, n).

    Returns
    -------
    A_align : np.ndarray
        Spectrally aligned {±1}-matrix.
    """

    # Project A onto the extremal spectral subspace
    PA = P_ext @ A

    # Entrywise sign map, zeros → +1
    A_align = np.where(PA >= 0, 1, -1)

    return A_align
