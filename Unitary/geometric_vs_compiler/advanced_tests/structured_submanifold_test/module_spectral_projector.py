# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# ============================================================
# MODULE 3: Spectral Projector
# File: module_spectral_projector.py
# ============================================================

import numpy as np
import cupy as cp
from module_deviation_operators import DeviationOperators


class SpectralProjector:
    """
    ============================================================
    ORIGINAL MATH (SIGNED ±1 MATRICES)
    ============================================================

        Δ(A) = A Aᵀ - n I

        Let Δ(A) = V Λ Vᵀ be the eigendecomposition.

        The extremal projector is:
            P_ext = Σ_{i ∈ J_ext} v_i v_iᵀ

        where J_ext indexes the dominant extremal modes:
            - largest |λ_i|
            - modes contributing most to T2 and M4
            - modes driving collapse geometry

        Purpose:
            Aligns updates with dominant spectral defects.
            Core operator in:
                - first-order extremal dynamics
                - second-order Hessian alignment
                - multiscale projector decomposition
                - collapse corridor geometry

    ============================================================
    UNITARY PORT (U ∈ U(n))
    ============================================================

        Δ(U) = U_target* U - I

        Δ(U) is not Hermitian in general.
        Therefore we use SVD:

            Δ(U) = U_svd Σ V_svd*

        Singular values σ_i replace eigenvalues λ_i.

        Dominant extremal modes correspond to:
            - largest |σ_i|
            - left singular vectors U_svd[:, i]

        Unitary extremal projector:
            P_ext^U = U_ext U_ext*

        where U_ext contains the top fraction of singular vectors.

        Purpose:
            Provides the geometric alignment direction for:
                - fidelity-driven optimization
                - extremal regularization
                - multiscale alignment
                - corridor navigation
    ============================================================
    """

    @staticmethod
    def projector_signed(A, frac_ext=0.8):
        """
        Original signed-matrix extremal projector.
        Uses eigenvectors of Δ(A).
        A is a numpy array.
        """
        Delta = DeviationOperators.delta_signed(A)

        lam, V = np.linalg.eigh(Delta)

        # Sort eigenvectors by descending |λ|
        idx = np.argsort(np.abs(lam))[::-1]
        V_sorted = V[:, idx]

        r = max(1, int(frac_ext * len(lam)))
        V_ext = V_sorted[:, :r]

        return V_ext @ V_ext.T

    @staticmethod
    def projector_unitary(U, U_target, frac_ext=0.8):
        """
        Unitary extremal projector.
        Uses left singular vectors of Δ(U).
        U and U_target are cupy arrays.
        """
        Delta = DeviationOperators.delta_unitary(U, U_target)

        # SVD of Δ(U)
        U_svd, sigma, _ = cp.linalg.svd(Delta)

        # Sort singular vectors by descending |σ|
        idx = cp.argsort(cp.abs(sigma))[::-1]
        U_sorted = U_svd[:, idx]

        r = max(1, int(frac_ext * len(sigma)))
        U_ext = U_sorted[:, :r]

        return U_ext @ U_ext.conj().T
