# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# ============================================================
# MODULE 5: Fidelity Objective
# File: module_fidelity_objective.py
# ============================================================

import cupy as cp
from module_extremal_invariant import ExtremalInvariant


class FidelityObjective:
    """
    ============================================================
    ORIGINAL MATH CONTEXT
    ============================================================

    In the signed ±1 collapse algebra, the objective is:
        - minimize deviation energy T2
        - minimize extremal invariant F(A)
        - enter collapse corridor
        - follow contraction laws

    None of these objectives attempt to match an external operator.

    ============================================================
    UNITARY PORT CONTEXT
    ============================================================

    When replacing Qiskit, the correct objective is:
        - maximize fidelity between U and U_target

    Two fidelity notions matter:

        1. State fidelity:
            Fid_state = |⟨ψ | U† U_target | ψ⟩|²
            (usually ψ = |0⟩)

        2. Full operator fidelity:
            Fid_op = |Tr(U† U_target)|² / N²
            (N = 2^n_qubits)

    These measure how close U is to U_target.

    ============================================================
    COMBINED OBJECTIVE
    ============================================================

    We combine fidelity with your extremal invariant:

        J(U) = - fidelity(U, U_target)
               + λ F_U(U)

    where:
        - fidelity drives U → U_target
        - F_U(U) regularizes deviation geometry
        - λ is a small weight (e.g., 0.1)

    This objective is what allowed your math to match Qiskit
    in the successful tests.
    ============================================================
    """

    # ------------------------------------------------------------
    # State fidelity
    # ------------------------------------------------------------
    @staticmethod
    def state_fidelity(U, U_target, n_qubits):
        """
        Compute state fidelity:
            Fid = |⟨0 | U† U_target | 0⟩|²
        U and U_target are cupy arrays.
        """
        N = 2**n_qubits

        psi0 = cp.zeros(N, dtype=cp.complex128)
        psi0[0] = 1.0

        psi_U = U @ psi0
        psi_T = U_target @ psi0

        return float(cp.abs(cp.vdot(psi_U, psi_T))**2)

    # ------------------------------------------------------------
    # Full operator fidelity
    # ------------------------------------------------------------
    @staticmethod
    def operator_fidelity(U, U_target):
        """
        Compute full operator fidelity:
            Fid_op = |Tr(U† U_target)|² / N²
        """
        N = U.shape[0]
        overlap = cp.trace(U.conj().T @ U_target)
        return float(cp.abs(overlap)**2 / (N**2))

    # ------------------------------------------------------------
    # Combined objective
    # ------------------------------------------------------------
    @staticmethod
    def combined(U, U_target, n_qubits, lam_reg=0.1):
        """
        Combined objective:
            J = - fidelity + λ F_U

        Returns:
            J, fidelity, extremal invariant
        """
        fid = FidelityObjective.state_fidelity(U, U_target, n_qubits)
        F_ext = ExtremalInvariant.extremal_unitary(U, U_target)

        J = -fid + lam_reg * F_ext

        return J, fid, F_ext
