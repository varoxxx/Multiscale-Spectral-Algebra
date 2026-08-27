# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# ============================================================
# MODULE 4: Re-Unitary Projection
# File: module_reunitary.py
# ============================================================

import numpy as np
import cupy as cp


class ReUnitary:
    """
    ============================================================
    ORIGINAL MATH (SIGNED ±1 MATRICES)
    ============================================================

        In collapse dynamics, after applying an extremal projector
        or block operator, the matrix is projected back into the
        discrete set {±1}^{n×n} using the sign operator:

            A_new = sign(M)

        This ensures:
            - A_new remains in the discrete manifold
            - collapse dynamics stay within the Hadamard search space
            - updates preserve the ±1 structure

        This is the discrete analogue of a projection back into the
        constraint manifold.

    ============================================================
    UNITARY PORT (U ∈ U(n))
    ============================================================

        For unitary matrices, the correct analogue of "sign(M)"
        is the *polar decomposition*:

            M = U H

        where:
            U is unitary
            H is Hermitian positive semidefinite

        The projection back into the unitary group is:

            U = M (M* M)^(-1/2)

        This is the closest unitary to M in Frobenius norm.

        Purpose:
            - ensures updates remain unitary
            - acts as the geometric analogue of sign(A)
            - preserves the constraint manifold U(n)
            - used after every projector-guided update
    ============================================================
    """

    @staticmethod
    def project(M):
        """
        Project a matrix M back into the unitary group using
        polar decomposition.
    
        M is a cupy array (complex matrix).
        Returns a cupy array U that is unitary.
        """
        # Compute M* M
        MM = M.conj().T @ M
    
        # Eigen-decomposition of M* M
        # CPU fallback for eigen-decomposition
        MM_cpu = cp.asnumpy(MM)
        lam_cpu, V_cpu = np.linalg.eigh(MM_cpu)
        
        lam = cp.asarray(lam_cpu)
        V = cp.asarray(V_cpu)
    
        # Inverse square root of eigenvalues
        lam_inv_sqrt = cp.where(lam > 1e-12, 1.0 / cp.sqrt(lam), 0.0)
    
        # Construct (M* M)^(-1/2)
        H_inv_sqrt = V @ cp.diag(lam_inv_sqrt) @ V.conj().T
    
        # Polar projection: U = M (M* M)^(-1/2)
        return M @ H_inv_sqrt
    
