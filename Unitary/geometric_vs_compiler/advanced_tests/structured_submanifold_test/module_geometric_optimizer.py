"""
David Mulnix copyright 2026
"""

# ============================================================
# MODULE 6: First-Order Geometric Optimizer
# File: module_geometric_optimizer.py
# ============================================================

import cupy as cp

from module_deviation_operators import DeviationOperators
from module_spectral_projector import SpectralProjector
from module_reunitary import ReUnitary
from module_fidelity_objective import FidelityObjective
from module_dynamical_invariants import DynamicalInvariants


class GeometricOptimizer:
    """
    Geometric optimizer with stall-aware spectral steering + early stopping.
    """

    @staticmethod
    def step(U, U_target, n_qubits, frac_ext=0.8, lam_reg=0.1, step_scale=0.1):
        """
        Perform ONE geometric update step.
        Returns (U_new, step_info)
        """
        Delta = DeviationOperators.delta_unitary(U, U_target)
        P_ext = SpectralProjector.projector_unitary(U, U_target, frac_ext)

        G = P_ext @ Delta
        G = G - lam_reg * Delta

        U_new = U - step_scale * G
        U_new = ReUnitary.project(U_new)

        return U_new, {"Delta_norm": float(cp.linalg.norm(Delta))}

    @staticmethod
    def optimize(
        U_init,
        U_target,
        n_qubits,
        steps=100,
        frac_ext=0.8,
        lam_reg=0.1,
        step_scale=0.1,
        verbose=False,
        eps_kappa=1e-5,
        eps_phi=1e-8,
        eps_m=-1e-6,
        eta_flow=0.1,
        stall_required=3,
        early_fid=1.0 - 1e-12,
        early_norm=1e-12,
        early_obj=1e-12,
    ):
        """
        Run first-order geometric optimization with stall-aware steering
        and early stopping conditions.

        Returns:
            U_final, history
        """

        U = U_init.copy()
        history = []

        invariants = DynamicalInvariants(beta=0.9)
        stall_counter = 0

        for k in range(steps):

            # Objective before update
            J, fid, F_ext = FidelityObjective.combined(
                U, U_target, n_qubits, lam_reg
            )
            history.append((J, fid, F_ext))

            if verbose:
                print(f"step {k:3d}  J={J:.6e}  fid={fid:.6e}  F_U={F_ext:.6e}")

            # Early stopping: perfect fidelity
            if fid >= early_fid:
                if verbose:
                    print("Early stopping: perfect fidelity reached.")
                break

            # Extremal projector
            P = SpectralProjector.projector_unitary(U, U_target, frac_ext)

            # Baseline geometric update
            U_tilde = (1.0 - step_scale) * U + step_scale * (P @ U_target)
            U_new = ReUnitary.project(U_tilde)

            # Stall diagnostics
            d_t, kappa_t, phi_t, m_t = invariants.update(U_new, U_target)

            if verbose:
                print(f"  diag: d={d_t:.6f} kappa={kappa_t:.3e} "
                      f"phi={phi_t:.3e} m={m_t:.3e}")

            stall = (
                abs(kappa_t) < eps_kappa and
                abs(phi_t)   < eps_phi   and
                m_t          < eps_m
            )

            if stall:
                stall_counter += 1
            else:
                stall_counter = 0

            # Spectral-flow kick
            if stall_counter >= stall_required:
                if verbose:
                    print("  stall detected -> applying spectral-flow kick")

                U_flow = U_new + eta_flow * (P @ U_new - U_new @ P)
                U_flow = ReUnitary.project(U_flow)

                # Accept kick if it improves J
                J_flow, _, _ = FidelityObjective.combined(
                    U_flow, U_target, n_qubits, lam_reg
                )
                J_new_no_kick = FidelityObjective.combined(
                    U_new, U_target, n_qubits, lam_reg
                )[0]

                if J_flow <= J_new_no_kick:
                    U_new = U_flow
                    if verbose:
                        print("  spectral-flow kick accepted")
                else:
                    if verbose:
                        print("  spectral-flow kick rejected")

                stall_counter = 0

            # Final objective check
            J_new, fid_new, _ = FidelityObjective.combined(
                U_new, U_target, n_qubits, lam_reg
            )

            if verbose:
                print(f"  J_new={J_new:.6e} fid_new={fid_new:.6e}")
                print("  ||U_new - U|| =", float(cp.linalg.norm(U_new - U)))

            # Early stopping: perfect fidelity after update
            if fid_new >= early_fid:
                if verbose:
                    print("Early stopping: perfect fidelity reached.")
                U = U_new
                break

            # Early stopping: update norm too small
            update_norm = float(cp.linalg.norm(U_new - U))
            if update_norm <= early_norm:
                if verbose:
                    print("Early stopping: update norm below threshold.")
                U = U_new
                break

            # Early stopping: objective improvement negligible
            if abs(J_new - J) <= early_obj:
                if verbose:
                    print("Early stopping: objective improvement negligible.")
                U = U_new
                break

            # ALWAYS ACCEPT UPDATE
            U = U_new
            if verbose:
                print("  update accepted")

        return U, history
