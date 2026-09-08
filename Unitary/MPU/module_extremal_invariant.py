# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""
# module_extremal_invariant.py

# ============================================================
# MODULE 2: Extremal Invariant
# File: module_extremal_invariant.py
# ============================================================

import numpy as np
import cupy as cp
from module_deviation_operators import DeviationOperators


class ExtremalInvariant:
    """
    ============================================================
    ORIGINAL MATH (SIGNED ±1 MATRICES)
    ============================================================

        Δ(A) = A Aᵀ - n I

        Let λ_i be eigenvalues of Δ(A).

        Quadratic spectral moment:
            T2(A) = Σ λ_i^2

        Quartic spectral moment:
            M4(A) = Σ λ_i^4

        Extremal invariant:
            F(A) = M4 / T2^2

        Purpose:
            Measures how "extremal" the deviation spectrum is.
            High F(A) = strong spectral defects.
            Low F(A) = collapse toward Hadamard structure.

    ============================================================
    UNITARY PORT (U ∈ U(n))
    ============================================================

        Δ(U) = U_target* U - I

        Δ(U) is not Hermitian in general, so we use singular values σ_i.

        Quadratic moment:
            T2(U) = Σ σ_i^2

        Quartic moment:
            M4(U) = Σ σ_i^4

        Unitary extremal invariant:
            F_U(U) = M4(U) / T2(U)^2

        Purpose:
            Measures deviation extremality of U relative to U_target.
            Used for:
                - geometric optimization
                - regularization
                - curvature analysis
                - corridor invariant
    ============================================================
    """

    # ------------------------------------------------------------
    # Signed-matrix extremal invariant (original math)
    # ------------------------------------------------------------
    @staticmethod
    def extremal_signed(A):
        """
        Compute F(A) for signed ±1 matrices using eigenvalues.
        A is a numpy array.
        """
        Delta = DeviationOperators.delta_signed(A)
        lam = np.linalg.eigvals(Delta)

        lam2 = lam**2
        lam4 = lam**4

        T2 = np.sum(lam2)
        M4 = np.sum(lam4)

        if T2 == 0:
            return 0.0

        return float(M4 / (T2**2))

    # ------------------------------------------------------------
    # Unitary extremal invariant (ported math)
    # ------------------------------------------------------------
    @staticmethod
    def extremal_unitary(U, U_target):
        """
        Compute F_U(U) using singular values of Δ(U).
        U and U_target are cupy arrays.
        """
        Delta = DeviationOperators.delta_unitary(U, U_target)

        # Singular values of Δ(U)
        sigma = cp.linalg.svd(Delta, compute_uv=False)

        sigma2 = sigma**2
        sigma4 = sigma**4

        T2 = float(cp.sum(sigma2))
        M4 = float(cp.sum(sigma4))

        if T2 == 0.0:
            return 0.0

        return float(M4 / (T2**2))
