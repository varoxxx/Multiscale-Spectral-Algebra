# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# ============================================================
# MODULE: Signature Distance and Spectral Templates
# File: module_signature_distance.py
# ============================================================

import cupy as cp
import numpy as np

from module_extremal_invariant import ExtremalInvariant  


class SignatureDistance:
    """
    Signature distance dist(U) in unitary space, plus optional
    lower-order spectral templates for blending.
    """

    @staticmethod
    def fluctuation_gram(U):
        """
        Δ(U) = U U* - I
        """
        n = U.shape[0]
        return U @ U.conj().T - cp.eye(n, dtype=cp.complex128)

    @staticmethod
    def eigenvalues_delta(U):
        """
        Eigenvalues of Δ(U) using CPU fallback because
        cp.linalg.eigvals is not available on many GPUs.
        """
        Delta = SignatureDistance.fluctuation_gram(U)
    
        # Move to CPU
        Delta_cpu = cp.asnumpy(Delta)
    
        # Compute eigenvalues on CPU
        vals_cpu = np.linalg.eigvals(Delta_cpu)
    
        # Move back to GPU
        vals = cp.asarray(vals_cpu)
    
        return vals


    @staticmethod
    def eigenvalue_spread(U):
        """
        Eigenvalue spread of Δ(U).
        """
        vals = SignatureDistance.eigenvalues_delta(U)
        re = cp.real(vals)
        return float(cp.max(re) - cp.min(re))

    @staticmethod
    def dist(
        U,
        U_target,
        w1=1.0,
        w2=1.0,
        w3=1.0,
        w4=1.0
    ):
        """
        Signature distance dist(U) relative to U_target.
        """
        # eigenvalues
        lam_U = SignatureDistance.eigenvalues_delta(U)
        lam_T = SignatureDistance.eigenvalues_delta(U_target)

        term1 = w1 * float(cp.sum(cp.abs(lam_U - lam_T)**2))

        # collapse invariant (extremal invariant)
        U_inv = ExtremalInvariant.extremal_unitary(U, U_target)
        T_inv = ExtremalInvariant.extremal_unitary(U_target, U_target)
        term2 = w2 * float(cp.abs(U_inv - T_inv))


        # Frobenius norm of Δ(U)
        Delta_U = SignatureDistance.fluctuation_gram(U)
        term3 = w3 * float(cp.linalg.norm(Delta_U))

        # eigenvalue spread
        spread_U = SignatureDistance.eigenvalue_spread(U)
        term4 = w4 * float(cp.abs(spread_U))

        return term1 + term2 + term3 + term4

    # --------------------------------------------------------
    # Optional spectral templates for blending
    # --------------------------------------------------------
    @staticmethod
    def template_eigenvalues(k):
        """
        Return eigenvalue template λ^{(k)} for k in {4,8,12}.
        Placeholder: user can replace with actual Hadamard-based templates.
        """
        if k == 4:
            # simple symmetric pattern
            return cp.array([2, 2, -2, -2], dtype=cp.complex128)
        elif k == 8:
            return cp.array([4, 4, 0, 0, -4, -4, 0, 0], dtype=cp.complex128)
        elif k == 12:
            return cp.array(
                [6, 6, 2, 2, -2, -2, -6, -6, 2, 2, -2, -2],
                dtype=cp.complex128
            )
        else:
            raise ValueError(f"Unsupported template size k={k}")

    @staticmethod
    def dist_template(U, k):
        """
        dist_k(U) = || λ(U) - λ^{(k)} ||^2
        """
        lam_U = SignatureDistance.eigenvalues_delta(U)
        lam_k = SignatureDistance.template_eigenvalues(k)
        # pad or truncate to match length
        n = lam_U.shape[0]
        m = lam_k.shape[0]
        if n == m:
            diff = lam_U - lam_k
        elif n < m:
            diff = lam_k[:n] - lam_U
        else:
            diff = lam_U[:m] - lam_k
        return float(cp.sum(cp.abs(diff)**2))
