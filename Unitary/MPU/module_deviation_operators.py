"""
David Mulnix copyright 2026
"""


# module_deviation_operators.py
# ============================================================
# MODULE 1: Deviation Operators
# ============================================================

import numpy as np
import cupy as cp

class DeviationOperators:
    """
    Original math:
        Δ(A) = A Aᵀ - n I
        Measures deviation from Hadamard structure.

    Unitary port:
        Δ(U) = U_target* U - I
        Measures deviation from target unitary.

    Purpose:
        Provides the core deviation operator used by:
            - extremal invariant
            - gradient
            - Hessian
            - multiscale decomposition
            - collapse corridor
    """

    @staticmethod
    def delta_signed(A):
        """
        Signed-matrix deviation operator.
        A is a numpy array (±1 entries).
        """
        n = A.shape[0]
        return A @ A.T - n * np.eye(n)

    @staticmethod
    def delta_unitary(U, U_target):
        """
        Unitary deviation operator.
        U and U_target are cupy arrays (complex unitary matrices).
        """
        return U_target.conj().T @ U - cp.eye(U.shape[0], dtype=cp.complex128)
