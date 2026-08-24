"""
David Mulnix copyright 2026

This script shows the translation of some of the math specficially the extremal from Multiscale Spectral Algebra.
It was an earlier stage exploration of the spectral geometry, it starts from a random matrix and uses random flips in a controlled mannner
to descened towards locating a valid hadamard. This is more of brute force search but it is also closed form algebra
showing that once a certain level of understanding is obtained with the algebra you can locate a valid hadamard.
Although the emperical testing and math has moved far beyond the limitations in this script it still should be helpful
to those who wish to see the math translated to funcational code. 

"""


import numpy as np

from module_fluctuation_gram import fluctuation_gram
from module_global_invariants import global_invariants
from module_extremal_invariant import spectral_extremal_invariant
from module_dominant_projector import dominant_extremal_projector
from module_alignment_functional import extremal_alignment_functional
from module_spectral_alignment import spectral_alignment
from module_strong_alignment import strong_spectral_alignment
from module_differential_objective import differential_collapse_objective
from module_move_set import (
    single_flip,
    block_flip_2x2,
    symmetry_breaking_flip,
)
from module_restart_supermartingale import restart_supermartingale


def extremal_collapse_n16(
    A_init,
    max_iters=2000,
    plateau_window=50,
    explore_steps=50,
    restart_loop_threshold=5,
):

    A = A_init.copy()
    n = A.shape[0]

    A_history = []
    F_history = []
    T2_history = []
    Spread_history = []

    best_F_so_far = np.inf
    last_improvement_iter = -1

    mode = "extremal"   # "extremal", "global", "explore"
    explore_counter = 0

    last_move = None
    last_state = None

    rng = np.random.default_rng()

    last_restart_index = None
    restart_repeat_count = 0

    for k in range(max_iters):

        Delta, lambdas, vecs = fluctuation_gram(A)
        invariants = global_invariants(Delta, lambdas)
        Spread = invariants["Spread"]
        F, M4, T2 = spectral_extremal_invariant(lambdas)

        print(f"\niter {k}: F={F:.6f}, T2={T2:.6f}, mode={mode}")

        if mode == "extremal" and F < best_F_so_far - 1e-12:
            best_F_so_far = F
            last_improvement_iter = k

        P_ext, J_ext, w = dominant_extremal_projector(lambdas, vecs)
        a_ext = extremal_alignment_functional(Delta, P_ext)
        objectives = differential_collapse_objective(
            Delta, lambdas, vecs, J_ext, F
        )
        D_spec = objectives["D_spec"]
        D_inv = objectives["D_inv"]
        D_mode = objectives["D_mode"]

        candidate_moves = []

        for i in range(n):
            for j in range(n):
                A_c = single_flip(A, i, j)
                _, lamb_c, _ = fluctuation_gram(A_c)
                F_c, _, T2_c = spectral_extremal_invariant(lamb_c)
                candidate_moves.append(("single", i, j, F_c, T2_c))

        for i in range(0, n - 1, 2):
            for j in range(0, n - 1, 2):
                A_c = block_flip_2x2(A, i, j)
                _, lamb_c, _ = fluctuation_gram(A_c)
                F_c, _, T2_c = spectral_extremal_invariant(lamb_c)
                candidate_moves.append(("block", i, j, F_c, T2_c))

        for i in range(n):
            for j in range(n):
                A_c = symmetry_breaking_flip(A, i, j)
                _, lamb_c, _ = fluctuation_gram(A_c)
                F_c, _, T2_c = spectral_extremal_invariant(lamb_c)
                candidate_moves.append(("sym", i, j, F_c, T2_c))

        best_move = None

        # --- EXTREMAL MODE ---
        if mode == "extremal":
            non_worsening = [m for m in candidate_moves if m[3] <= F + 1e-12]
            in_plateau = (k - last_improvement_iter) >= plateau_window

            if non_worsening and not in_plateau:
                best_F = np.inf
                best_T2 = np.inf
                for move_type, i, j, F_c, T2_c in non_worsening:
                    if F_c < best_F - 1e-12 or (
                        abs(F_c - best_F) < 1e-12 and T2_c < best_T2 - 1e-12
                    ):
                        best_F = F_c
                        best_T2 = T2_c
                        best_move = (move_type, i, j, F_c, T2_c)
            else:
                if T2 > 1e-8:
                    print("  SWITCHING TO GLOBAL MODE")
                    mode = "global"

        # --- GLOBAL MODE ---
        if mode == "global":
            strict_moves = []
            for move_type, i, j, F_c, T2_c in candidate_moves:
                if T2_c >= T2 - 1e-12:
                    continue
                if last_move is not None and (move_type, i, j) == last_move:
                    continue
                if last_state is not None:
                    F_prev, T2_prev = last_state
                    if abs(F_c - F_prev) < 1e-12 and abs(T2_c - T2_prev) < 1e-12:
                        continue
                strict_moves.append((move_type, i, j, F_c, T2_c))

            if strict_moves:
                best_T2 = np.inf
                best_F = np.inf
                for move_type, i, j, F_c, T2_c in strict_moves:
                    if T2_c < best_T2 - 1e-12 or (
                        abs(T2_c - best_T2) < 1e-12 and F_c < best_F - 1e-12
                    ):
                        best_T2 = T2_c
                        best_F = F_c
                        best_move = (move_type, i, j, F_c, T2_c)
            else:
                print("  GLOBAL STUCK (no T2 descent) → ENTERING EXPLORE MODE")
                mode = "explore"
                explore_counter = 0
                best_move = None

        # --- EXPLORE MODE ---
        if mode == "explore":
            explore_counter += 1
            exploratory_moves = []
            for move_type, i, j, F_c, T2_c in candidate_moves:
                if last_move is not None and (move_type, i, j) == last_move:
                    continue
                if last_state is not None:
                    F_prev, T2_prev = last_state
                    if abs(F_c - F_prev) < 1e-12 and abs(T2_c - T2_prev) < 1e-12:
                        continue
                exploratory_moves.append((move_type, i, j, F_c, T2_c))

            if exploratory_moves:
                exploratory_moves.sort(key=lambda m: (m[4], m[3]))
                top_k = min(10, len(exploratory_moves))
                best_move = exploratory_moves[rng.integers(0, top_k)]
            else:
                print("  EXPLORE MODE: no moves available → STOP")
                break

            if explore_counter >= explore_steps:
                print("  EXPLORE BUDGET EXHAUSTED → BACK TO GLOBAL (restart temporarily disabled)")
                mode = "global"
                restart_repeat_count = -999  # disable restart for next extremal entry


        if best_move is None:
            print("  NO MOVE AVAILABLE → STOP")
            break

        move_type, i_best, j_best, best_F_c, best_T2_c = best_move
        print(
            f"  chosen: {move_type}({i_best},{j_best}) "
            f"F_c={best_F_c:.6f}, T2_c={best_T2_c:.6f}"
        )

        if move_type == "single":
            A = single_flip(A, i_best, j_best)
        elif move_type == "block":
            A = block_flip_2x2(A, i_best, j_best)
        elif move_type == "sym":
            A = symmetry_breaking_flip(A, i_best, j_best)

        if mode in ("global", "explore"):
            last_move = (move_type, i_best, j_best)
            last_state = (F, T2)

        Delta_c, lamb_c, _ = fluctuation_gram(A)
        F_c, _, T2_c = spectral_extremal_invariant(lamb_c)
        Spread_c = global_invariants(Delta_c, lamb_c)["Spread"]

        print(f"  AFTER MOVE: F_c={F_c:.6f}, T2_c={T2_c:.6f}")

        if mode == "extremal":
            if F_c < F - 1e-12:
                print("  SKIP ALIGN (improved F)")
            elif abs(F_c) < 1e-12 and abs(T2_c) < 1e-12:
                print("  SKIP ALIGN (Hadamard)")
            else:
                if a_ext > 0:
                    A = spectral_alignment(A, P_ext)
                if (
                    D_mode > 1.5 * D_spec
                    or D_inv > 1.25 * D_spec
                    or Spread > 1.0 * np.sqrt(max(T2, 1e-12))
                ):
                    A = strong_spectral_alignment(A, P_ext)

                Delta_c, lamb_c, _ = fluctuation_gram(A)
                F_c, _, T2_c = spectral_extremal_invariant(lamb_c)
                Spread_c = global_invariants(Delta_c, lamb_c)["Spread"]
                print(f"  AFTER ALIGN: F_c={F_c:.6f}, T2_c={T2_c:.6f}")

        A_history.append(A.copy())
        F_history.append(F_c)
        T2_history.append(T2_c)
        Spread_history.append(Spread_c)

        if mode == "extremal":

            # --- restart suppression after explore ---
            if restart_repeat_count < 0:
                print("  RESTART DISABLED FOR ONE ITERATION AFTER EXPLORE")
                restart_repeat_count = 0
                continue  # skip restart entirely

            A_restart, j_restart = restart_supermartingale(
                A_history,
                F_history,
                T2_history=T2_history,
                Spread_history=Spread_history,
            )

            # anti-loop on restart index
            if last_restart_index is not None and j_restart == last_restart_index:
                restart_repeat_count += 1
            else:
                restart_repeat_count = 0
                last_restart_index = j_restart

            if restart_repeat_count >= restart_loop_threshold:
                # pick a random history index instead of the same j_restart
                j_alt = rng.integers(0, len(A_history))
                print(
                    f"  RESTART LOOP DETECTED (j={j_restart} repeated) → "
                    f"USING RANDOM j={j_alt}"
                )
                A_restart = A_history[j_alt].copy()
                j_restart = j_alt

            print(
                f"  RESTART → j={j_restart}, "
                f"F_j={F_history[j_restart]:.6f}, "
                f"T2_j={T2_history[j_restart]:.6f}"
            )
            A = A_restart.copy()
        else:
            print("  NON-EXTREMAL MODE: no restart")

        if abs(T2_c) < 1e-8 and abs(F_c) < 1e-8:
            print("  REACHED HADAMARD BASIN")
            break

    return A, {
        "A": A_history,
        "F": F_history,
        "T2": T2_history,
        "Spread": Spread_history,
    }


if __name__ == "__main__":
    A0 = np.random.choice([-1, 1], size=(16, 16))
    A_final, history = extremal_collapse_n16(A0, max_iters=2000)

    print("\nInitial A0:")
    print(A0)
    print("\nFinal A_final:")
    print(A_final)

    with open("final_matrix_n16.txt", "w") as f:
        for row in A_final:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

    Delta_final, lamb_final, _ = fluctuation_gram(A_final)
    F_final, M4_final, T2_final = spectral_extremal_invariant(lamb_final)

    print("\nFinal invariants:")
    print("F =", F_final)
    print("T2 =", T2_final)
    print("Spread =", lamb_final.max() - lamb_final.min())
    print("‖Δ‖ =", np.linalg.norm(Delta_final))
