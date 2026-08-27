# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

# ============================================================
# MODULE: Dynamical Invariants (κ_t, φ_t, m_t)
# File: module_dynamical_invariants.py
# ============================================================

import cupy as cp

from module_signature_distance import SignatureDistance


class DynamicalInvariants:
    """
    Track and compute curvature κ_t, spectral flow φ_t, and
    momentum m_t for a unitary trajectory U_t.
    """

    def __init__(self, beta=0.9):
        self.beta = beta
        self.d_history = []      # scalar distances d_t
        self.lambda_history = [] # eigenvalues of Δ(U_t)
        self.m_t = 0.0

    def update(self, U, U_target):
        """
        Update invariants given current U and U_target.
        Returns (d_t, κ_t, φ_t, m_t).
        """
        # distance
        d_t = SignatureDistance.dist(U, U_target)
        self.d_history.append(d_t)

        # eigenvalues
        lam_t = SignatureDistance.eigenvalues_delta(U)
        self.lambda_history.append(lam_t)

        # curvature κ_t
        if len(self.d_history) >= 3:
            d0 = self.d_history[-1]
            d1 = self.d_history[-2]
            d2 = self.d_history[-3]
            kappa_t = d0 - 2 * d1 + d2
        else:
            kappa_t = 0.0

        # spectral flow φ_t
        if len(self.lambda_history) >= 2:
            lam_curr = self.lambda_history[-1]
            lam_prev = self.lambda_history[-2]
            # pad/truncate to match
            n = lam_curr.shape[0]
            m = lam_prev.shape[0]
            if n == m:
                diff = lam_curr - lam_prev
            elif n < m:
                diff = lam_prev[:n] - lam_curr
            else:
                diff = lam_curr[:m] - lam_prev
            phi_t = float(cp.mean(cp.real(diff)))

        else:
            phi_t = 0.0

        # momentum m_t
        if len(self.d_history) >= 2:
            d_curr = self.d_history[-1]
            d_prev = self.d_history[-2]
            delta_d = d_curr - d_prev
            self.m_t = self.beta * self.m_t + (1.0 - self.beta) * delta_d
        # else m_t stays as initialized

        return d_t, kappa_t, phi_t, self.m_t
