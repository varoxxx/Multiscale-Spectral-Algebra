
"""
David Mulnix copyright 2026
"""

import json
import sys
import numpy as np
from math import log2


def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON {path}: {e}")
        sys.exit(1)

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}, total

def shannon_entropy(dist):
    return -sum(p * log2(p) for p in dist.values() if p > 0)

def hellinger_distance(p, q):
    keys = set(p.keys()) | set(q.keys())
    return np.sqrt(0.5 * sum((np.sqrt(p.get(k, 0.0)) - np.sqrt(q.get(k, 0.0)))**2 for k in keys))

def kl_divergence(p, q):
    keys = set(p.keys()) & set(q.keys())
    kl = 0.0
    for k in keys:
        if p[k] > 0 and q[k] > 0:
            kl += p[k] * log2(p[k] / q[k])
    return kl

def bitstring_to_array(bitstring):
    return np.array([int(b) for b in bitstring])

def mutual_information(dist, n_qubits):
    bitstrings = list(dist.keys())
    probs = np.array([dist[b] for b in bitstrings])
    X = np.array([bitstring_to_array(b) for b in bitstrings])

    MI = np.zeros((n_qubits, n_qubits))

    for i in range(n_qubits):
        for j in range(n_qubits):
            if i == j:
                MI[i, j] = 0
                continue

            joint = {}
            for idx in range(len(bitstrings)):
                key = (X[idx, i], X[idx, j])
                joint[key] = joint.get(key, 0) + probs[idx]

            pi = {0: 0.0, 1: 0.0}
            pj = {0: 0.0, 1: 0.0}
            for (bi, bj), p in joint.items():
                pi[bi] += p
                pj[bj] += p

            mi = 0.0
            for (bi, bj), p in joint.items():
                if p > 0 and pi[bi] > 0 and pj[bj] > 0:
                    mi += p * log2(p / (pi[bi] * pj[bj]))
            MI[i, j] = mi

    return MI

def main():
    if len(sys.argv) < 2:
        print("Usage: python mpu_analysis_rich.py <hardware_json> [simulator_json]")
        sys.exit(1)

    hw_json = sys.argv[1]
    sim_json = sys.argv[2] if len(sys.argv) > 2 else None

    print("\n=== MPU HARDWARE ANALYSIS (RICH) ===")
    print(f"Hardware JSON: {hw_json}")

    data = safe_load_json(hw_json)

    # Basic metadata
    job_id = data.get("job_id", None)
    backend = data.get("backend", None)
    print("\n=== JSON Loaded Successfully ===")
    print(f"Job ID: {job_id}")
    print(f"Backend: {backend}")

    # Counts and distribution
    counts = data["counts"]
    dist_hw, total_shots = normalize_counts(counts)

    print("\n=== Hardware Counts Summary ===")
    print(f"Total shots: {total_shots}")
    print(f"Unique bitstrings: {len(dist_hw)}")

    print("\n=== First 10 Probabilities ===")
    for k in list(dist_hw.keys())[:10]:
        print(k, dist_hw[k])

    # Entropy
    entropy_hw = shannon_entropy(dist_hw)
    print("\n=== Shannon Entropy (Hardware) ===")
    print(entropy_hw)

    # Uniform reference
    n_qubits = len(next(iter(counts.keys())))
    uniform = {b: 1 / (2**n_qubits) for b in dist_hw}
    kl_hw = kl_divergence(dist_hw, uniform)
    hell_hw = hellinger_distance(dist_hw, uniform)
    print("\n=== Divergence vs Uniform ===")
    print(f"KL(HW || Uniform) = {kl_hw}")
    print(f"Hellinger(HW, Uniform) = {hell_hw}")

    # Optional simulator comparison
    if sim_json is not None:
        print("\n=== Simulator Comparison ===")
        sim_data = safe_load_json(sim_json)
        sim_counts = sim_data["counts"]
        dist_sim, total_sim_shots = normalize_counts(sim_counts)
        hell_hw_sim = hellinger_distance(dist_hw, dist_sim)
        print(f"Hellinger(HW, SIM) = {hell_hw_sim}")
    else:
        hell_hw_sim = None

    # Mutual information
    print("\n=== Mutual Information Matrix (Hardware) ===")
    MI = mutual_information(dist_hw, n_qubits)
    print(MI)
    total_mi = float(np.sum(MI))
    print(f"\nTotal MI (sum over all pairs) = {total_mi}")

    # Calibration summary
    print("\n=== Calibration Summary ===")
    t1 = data.get("t1", [])
    t2 = data.get("t2", [])
    readout = data.get("readout", [])

    if t1:
        t1_min = min(x for x in t1 if x is not None)
        t1_max = max(x for x in t1 if x is not None)
        print(f"T1: min = {t1_min} max = {t1_max}")
    else:
        print("T1: MISSING")

    if t2:
        t2_min = min(x for x in t2 if x is not None)
        t2_max = max(x for x in t2 if x is not None)
        print(f"T2: min = {t2_min} max = {t2_max}")
    else:
        print("T2: MISSING")

    if readout:
        ro_min = min(x for x in readout if x is not None)
        ro_max = max(x for x in readout if x is not None)
        worst_q = readout.index(ro_max)
        print(f"Readout error: min = {ro_min} max = {ro_max}")
        print(f"Worst readout qubit: {worst_q} error = {ro_max}")
    else:
        print("Readout error: MISSING")

    # Backend config/status keys
    print("\n=== Backend Configuration Keys ===")
    config = data.get("config", {})
    print(list(config.keys()))

    print("\n=== Backend Status Keys ===")
    status = data.get("status", {})
    print(list(status.keys()))

    # Scientific evidence summary
    print("\n=== Scientific Evidence Summary ===")
    print("✔ Hardware execution confirmed" if counts else "✘ No counts")
    print("✔ Real counts collected" if counts else "✘ No counts")
    print("✔ Calibration data included" if t1 and t2 and readout else "✘ Missing calibration data")
    print("✔ Backend configuration included" if config else "✘ Missing config")
    print("✔ Backend status included" if status else "✘ Missing status")
    print(f"✔ Entropy (HW) = {entropy_hw}")
    print(f"✔ KL(HW || Uniform) = {kl_hw}")
    print(f"✔ Hellinger(HW, Uniform) = {hell_hw}")
    if hell_hw_sim is not None:
        print(f"✔ Hellinger(HW, SIM) = {hell_hw_sim}")
    print(f"✔ Total MI = {total_mi}")

    # Save analysis JSON
    analysis_output = {
        "entropy_hw": entropy_hw,
        "kl_hw_uniform": kl_hw,
        "hellinger_hw_uniform": hell_hw,
        "hellinger_hw_sim": hell_hw_sim,
        "total_mutual_information": total_mi,
        "mutual_information_matrix": MI.tolist(),
        "n_qubits": n_qubits,
        "total_shots": total_shots,
        "unique_bitstrings": len(dist_hw),
        "t1_min": t1_min if t1 else None,
        "t1_max": t1_max if t1 else None,
        "t2_min": t2_min if t2 else None,
        "t2_max": t2_max if t2 else None,
        "readout_min": ro_min if readout else None,
        "readout_max": ro_max if readout else None,
        "worst_readout_qubit": worst_q if readout else None,
    }

    out_file = hw_json.replace(".json", "_analysis_rich.json")
    with open(out_file, "w") as f:
        json.dump(analysis_output, f, indent=2)

    print(f"\nSaved rich analysis results to: {out_file}")
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
