# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# module_move_set.py

import numpy as np

def single_flip(A: np.ndarray, i: int, j: int):
    """
    Single flip: A_ij -> -A_ij
    """
    B = A.copy()
    B[i, j] = -B[i, j]
    return B

def block_flip_2x2(A: np.ndarray, i: int, j: int):
    """
    2x2 block flip:
        flip entries in the block [i:i+2, j:j+2]
    """
    B = A.copy()
    B[i:i+2, j:j+2] = -B[i:i+2, j:j+2]
    return B

def gradient_guided_score(Delta: np.ndarray, grad_F: np.ndarray, delta_update: np.ndarray):
    """
    Gradient-guided score using <δ, ∇F(A)>_F.
    """
    return float(np.sum(delta_update * grad_F))

def projector_guided_score(delta_update: np.ndarray, P_ext: np.ndarray):
    """
    Projector-guided score using <δ, P_ext>_F.
    """
    return float(np.sum(delta_update * P_ext))

def flip_delta_update(A: np.ndarray, i: int, j: int):
    """
    Compute the induced Δ(A) update δ for a single flip at (i, j).

    A flip at (i, j) changes row i and column j; the induced update
    in Δ(A) is rank-2. Here we compute δ explicitly:

        A' = A with A_ij flipped
        Δ' = A' A'^T - n I
        δ  = Δ' - Δ

    This is used for scoring moves via gradient or projector guidance.
    """
    n = A.shape[0]
    # Original Δ(A)
    Delta = A @ A.T - n * np.eye(n)
    # Flipped matrix
    A_flip = single_flip(A, i, j)
    Delta_flip = A_flip @ A_flip.T - n * np.eye(n)
    delta = Delta_flip - Delta
    return delta

def symmetry_breaking_flip(A: np.ndarray, i: int, j: int):
    """
    Symmetry-breaking flip: same as single flip, but conceptually
    reserved for moves that escape stall basins.
    """
    return single_flip(A, i, j)
