"""
David Mulnix copyright 2026
"""

import json
import numpy as np

# ============================================================
# Core math utilities (with corrected bit ordering)
# ============================================================

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def bitstring_to_array(bitstring):
    # Reverse bitstring so index i corresponds to physical qubit i
    return np.array([int(b) for b in reversed(bitstring)], dtype=np.int32)

def shannon_entropy(p):
    return -sum(v * np.log2(v) for v in p.values() if v > 0)

def mutual_information(p, i, j):
    joint = np.zeros((2, 2))
    for bitstring, prob in p.items():
        b = bitstring_to_array(bitstring)
        joint[b[i], b[j]] += prob
    pi = joint.sum(axis=1)
    pj = joint.sum(axis=0)
    mi = 0.0
    for a in range(2):
        for c in range(2):
            if joint[a, c] > 0:
                mi += joint[a, c] * np.log2(joint[a, c] / (pi[a] * pj[c]))
    return mi

def correlation(p, i, j):
    xs, ys, ws = [], [], []
    for bitstring, prob in p.items():
        b = bitstring_to_array(bitstring)
        xs.append(b[i])
        ys.append(b[j])
        ws.append(prob)
    xs = np.array(xs)
    ys = np.array(ys)
    ws = np.array(ws)
    Ex = np.sum(xs * ws)
    Ey = np.sum(ys * ws)
    Exy = np.sum(xs * ys * ws)
    return Exy - Ex * Ey

# ============================================================
# Load JSON helper
# ============================================================

def load_counts(filename):
    with open(filename, "r") as f:
        return json.load(f)["counts"]

# ============================================================
# Correct expected behaviors for baselines
# ============================================================

def expected_entropy(gate, state, N=6):
    # Uniform over 2^N for allplus
    if state == "allplus":
        return float(N)

    # For plusplus, plus0, 0plus:
    # Only first two qubits nontrivial
    if state == "plusplus":
        return 2.0  # 4 outcomes
    if state in ("plus0", "0plus"):
        return 1.0  # 2 outcomes

    return None

def expected_pair_behavior(gate, state):
    """
    Expected MI and Corr between physical qubits (0,1).
    These expectations now match actual physics.
    """

    # -------------------------
    # CZ baseline expectations
    # -------------------------
    if gate == "cz":
        if state == "plusplus":
            # CZ on |++> gives uniform distribution -> no correlation
            return 0.0, 0.0
        if state == "plus0":
            return 0.0, 0.0
        if state == "0plus":
            return 0.0, 0.0
        if state == "allplus":
            return 0.0, 0.0

    # -------------------------
    # CNOT baseline expectations
    # -------------------------
    if gate == "cnot":
        if state == "plusplus":
            # CNOT leaves |++> uniform -> no correlation
            return 0.0, 0.0
        if state == "plus0":
            # Bell pair
            return 1.0, 0.25
        if state == "0plus":
            return 0.0, 0.0
        if state == "allplus":
            return 0.0, 0.0

    # -------------------------
    # QFT-LIKE baseline expectations
    # -------------------------
    if gate == "qft_like":
        # CX chain behavior in Z basis:
        if state == "plus0":
            return "nonzero", "nonzero"   # strong GHZ-like correlation
        if state == "plusplus":
            return "small", "small"       # very weak correlation in Z
        if state == "0plus":
            return 0.0, 0.0               # product
        if state == "allplus":
            return "small", "small"       # small residual correlation


    return None, None

# ============================================================
# Sanity check runner
# ============================================================

def sanity_check_baseline(N=6):
    gates = ["cz", "cnot", "qft_like"]
    states = ["plusplus", "plus0", "0plus", "allplus"]

    overall_ok = True

    for gate in gates:
        for state in states:
            fname = f"{gate}_{state}.json"
            counts = load_counts(fname)
            p = normalize_counts(counts)

            H = shannon_entropy(p)
            H_exp = expected_entropy(gate, state, N)
            H_ok = (H_exp is None) or (abs(H - H_exp) < 0.05)

            mi_01 = mutual_information(p, 0, 1)
            corr_01 = correlation(p, 0, 1)
            mi_exp, corr_exp = expected_pair_behavior(gate, state)

            # MI check
            if mi_exp == "nonzero":
                mi_ok = mi_01 > 0.1
            elif mi_exp == "small":
                mi_ok = abs(mi_01) < 0.05
            elif isinstance(mi_exp, float):
                mi_ok = abs(mi_01 - mi_exp) < 0.05
            else:
                mi_ok = True

            # Corr check
            if corr_exp == "nonzero":
                corr_ok = abs(corr_01) > 0.05
            elif corr_exp == "small":
                corr_ok = abs(corr_01) < 0.05
            elif isinstance(corr_exp, float):
                corr_ok = abs(corr_01 - corr_exp) < 0.02
            else:
                corr_ok = True

            print(f"=== Sanity check: {gate.upper()} |{state}⟩ ===")
            print(f"Entropy: H={H:.4f}, expected={H_exp}, OK={H_ok}")
            print("MI/Corr details for key pairs:")
            print(f"  pair (0,1): MI={mi_01:.4f}, Corr={corr_01:.4f}, "
                  f"expected MI={mi_exp}, Corr={corr_exp}")
            print(f"MI OK={mi_ok}, Corr OK={corr_ok}\n")

            if not (H_ok and mi_ok and corr_ok):
                overall_ok = False

    print("====================================================")
    print(f"Overall baseline sanity: {'PASS' if overall_ok else 'FAIL'}")
    print("====================================================")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    sanity_check_baseline(N=6)
