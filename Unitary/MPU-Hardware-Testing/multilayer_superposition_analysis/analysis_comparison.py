"""
David Mulnix copyright 2026
"""

import json
import numpy as np

# ============================================================
# Core math utilities
# ============================================================

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def bitstring_to_array(bitstring):
    s = bitstring.replace(" ", "")
    return np.array([int(b) for b in s], dtype=np.int32)

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
        for b in range(2):
            if joint[a, b] > 0:
                mi += joint[a, b] * np.log2(joint[a, b] / (pi[a] * pj[b]))
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

def hellinger(p, q):
    keys = set(p.keys()) | set(q.keys())
    return np.sqrt(0.5 * sum(
        (np.sqrt(p.get(k, 0)) - np.sqrt(q.get(k, 0)))**2 for k in keys
    ))

def collapse_geometry_signature(p):
    keys = sorted(p.keys())
    vec = np.array([p[k] for k in keys])
    centered = vec - vec.mean()
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 0 else centered

def collapse_geometry_signature_aligned(p, keys_universe):
    vec = np.array([p.get(k, 0.0) for k in keys_universe])
    centered = vec - vec.mean()
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 0 else centered

# ============================================================
# Bit‑alignment utilities (MI‑based)
# ============================================================

def mi_matrix(counts, N):
    p = normalize_counts(counts)
    MI = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                MI[i, j] = mutual_information(p, i, j)
    return MI

def compute_mapping(MI_mpu, MI_base):
    N = MI_mpu.shape[0]
    mapping = {}
    used = set()
    for i in range(N):
        best_j = None
        best_score = -1
        for j in range(N):
            if j in used:
                continue
            v_mpu = MI_mpu[i].copy()
            v_base = MI_base[j].copy()
            v_mpu[i] = 0
            v_base[j] = 0
            score = np.dot(v_mpu, v_base)
            if score > best_score:
                best_score = score
                best_j = j
        mapping[i] = best_j
        used.add(best_j)
    return mapping

def apply_qubit_mapping(counts, mapping):
    remapped = {}
    for bitstring, v in counts.items():
        b = bitstring_to_array(bitstring)
        new_b = np.zeros_like(b)
        for old_q, new_q in mapping.items():
            new_b[new_q] = b[old_q]
        new_key = "".join(str(x) for x in new_b)
        remapped[new_key] = remapped.get(new_key, 0) + v
    return remapped

# ============================================================
# Analysis function
# ============================================================

def analyze_test(label, input_state, counts):
    print("\n====================================================")
    print("=== Analysis for {label} (Input state: {input_state}) ===")
    print("====================================================")

    counts_clean = {k.replace(" ", ""): v for k, v in counts.items()}
    p = normalize_counts(counts_clean)

    H = shannon_entropy(p)
    print(f"Shannon entropy: {H:.4f}")

    N = len(next(iter(counts_clean.keys())))
    MI = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                MI[i, j] = mutual_information(p, i, j)
    print("\nMutual information (MI) matrix:")
    print(MI)

    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                C[i, j] = correlation(p, i, j)
    print("\nCorrelation matrix:")
    print(C)

    sig = collapse_geometry_signature(p)
    print("\nCollapse geometry signature (normalized):")
    print(sig)

    return p, MI, C, sig

# ============================================================
# Load JSON helper
# ============================================================

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)["counts"]

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ASCII‑safe state labels
    # ============================================================

    input_states = ["plusplus", "plus0", "0plus", "allplus"]

    mpu_files = {
        "plusplus": "ibm_run_multilayer_N6_test_case_1.json",
        "plus0": "ibm_run_multilayer_N6_test_case_2.json",
        "0plus": "ibm_run_multilayer_N6_test_case_3.json",
        "allplus": "ibm_run_multilayer_N6_default.json"
    }

    baseline_files = {
        "plusplus": "cz_plusplus.json",
        "plus0": "cz_plus0.json",
        "0plus": "cz_0plus.json",
        "allplus": "cz_allplus.json"
    }

    # ============================================================
    # Step 1: Compute mapping using structured states
    # ============================================================

    print("\n=== BITWISE ALIGNMENT CHECK ===")

    structured_states = ["plusplus", "plus0", "0plus"]
    mapping_identity = {i: i for i in range(6)}

    for state in structured_states:
        mpu_counts = load_json(mpu_files[state])
        base_counts = load_json(baseline_files[state])
        MI_mpu = mi_matrix(mpu_counts, 6)
        MI_base = mi_matrix(base_counts, 6)
        mapping = compute_mapping(MI_mpu, MI_base)
        print(f"{state}: {mapping}")

    print("\nUsing forced identity mapping for all analysis.\n")

    # ============================================================
    # Step 2: Run analysis with aligned counts
    # ============================================================

    mpu_results = {}
    for state in input_states:
        counts = load_json(mpu_files[state])
        aligned_counts = apply_qubit_mapping(counts, mapping_identity)
        p, MI, C, sig = analyze_test(f"MPU {state}", state, aligned_counts)
        mpu_results[state] = (p, MI, C, sig)

    baselines = ["cz", "cnot", "qft_like"]
    baseline_results = {}

    for base in baselines:
        baseline_results[base] = {}
        for state in input_states:
            fname = f"{base}_{state}.json"
            counts = load_json(fname)
            aligned_counts = apply_qubit_mapping(counts, mapping_identity)
            p, MI, C, sig = analyze_test(f"{base.upper()} {state}", state, aligned_counts)
            baseline_results[base][state] = (p, MI, C, sig)

    # ============================================================
    # Step 3: MPU vs Baseline comparisons
    # ============================================================

    print("\n====================================================")
    print("=== MPU vs Baseline Comparisons (Hellinger + Geometry) ===")
    print("====================================================")

    for base in baselines:
        print(f"\n******** {base.upper()} ********")
        for state in input_states:
            p_mpu, _, _, _ = mpu_results[state]
            p_base, _, _, _ = baseline_results[base][state]

            keys_universe = sorted(set(p_mpu.keys()) | set(p_base.keys()))

            sig_mpu = collapse_geometry_signature_aligned(p_mpu, keys_universe)
            sig_base = collapse_geometry_signature_aligned(p_base, keys_universe)

            H = hellinger(p_mpu, p_base)
            dot = np.dot(sig_mpu, sig_base)

            print(f"{state}:  Hellinger = {H:.4f},   Geometry dot = {dot:.4f}")

    # ============================================================
    # Step 4: Dual comparison for ALLPLUS
    # ============================================================

    print("\n====================================================")
    print("=== ALLPLUS Dual Comparison (Default + Forced Identity) ===")
    print("====================================================")

    state = "allplus"
    p_mpu, _, _, _ = mpu_results[state]

    for base in baselines:
        p_base, _, _, _ = baseline_results[base][state]
        keys_universe = sorted(set(p_mpu.keys()) | set(p_base.keys()))

        sig_mpu = collapse_geometry_signature_aligned(p_mpu, keys_universe)
        sig_base = collapse_geometry_signature_aligned(p_base, keys_universe)

        H = hellinger(p_mpu, p_base)
        dot = np.dot(sig_mpu, sig_base)

        print(f"{base.upper()} ALLPLUS (forced identity):  Hellinger = {H:.4f}, Geometry dot = {dot:.4f}")
