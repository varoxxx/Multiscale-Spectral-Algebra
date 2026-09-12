"""
David Mulnix copyright 2026
"""
import json
import numpy as np
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict
import sys
from math import log2

# ------------------------------------------------------------
# Load JSON safely
# ------------------------------------------------------------
def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON {path}: {e}")
        sys.exit(1)

# ------------------------------------------------------------
# Extract probabilities from JSON
# ------------------------------------------------------------
def extract_probabilities(data):
    counts = data.get("counts", {})
    if not counts:
        print("ERROR: JSON missing 'counts'")
        sys.exit(1)

    shots = data.get("total_shots", None)
    if shots is None:
        shots = sum(counts.values())

    probs = {b: counts[b] / shots for b in counts}
    return probs, shots

# ------------------------------------------------------------
# Compute MI if missing
# ------------------------------------------------------------
def bitstring_to_array(bitstring):
    return np.array([int(b) for b in bitstring])

def compute_mutual_information(probs):
    bitstrings = list(probs.keys())
    n_qubits = len(bitstrings[0])
    pvals = np.array([probs[b] for b in bitstrings])
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
                joint[key] = joint.get(key, 0) + pvals[idx]

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

# ------------------------------------------------------------
# Probability Clustering
# ------------------------------------------------------------
def cluster_probabilities(probs, num_clusters=4):
    values = np.array(list(probs.values()))
    quantiles = np.quantile(values, np.linspace(0, 1, num_clusters+1))

    clusters = defaultdict(list)
    for bitstring, p in probs.items():
        for i in range(num_clusters):
            if quantiles[i] <= p <= quantiles[i+1]:
                clusters[i].append(bitstring)
                break
    return clusters, quantiles

def cluster_stats(probs, clusters):
    stats = {}
    for cid, bitstrings in clusters.items():
        vals = [probs[b] for b in bitstrings]
        stats[cid] = {
            "count": len(bitstrings),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "entropy": float(-np.sum([p*np.log2(p) for p in vals if p > 0]))
        }
    return stats

# ------------------------------------------------------------
# MI Spectral Analysis
# ------------------------------------------------------------
def mi_spectral_analysis(mi_matrix):
    eigs = np.linalg.eigvals(mi_matrix)
    eigs = np.real(eigs)

    spectral_radius = np.max(np.abs(eigs))
    participation_ratio = (np.sum(eigs)**2) / np.sum(eigs**2)

    return {
        "eigenvalues": eigs.tolist(),
        "spectral_radius": float(spectral_radius),
        "participation_ratio": float(participation_ratio)
    }

# ------------------------------------------------------------
# MI vs Distance
# ------------------------------------------------------------
def mi_vs_distance(mi_matrix):
    N = mi_matrix.shape[0]
    profile = defaultdict(list)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = abs(i - j)
                profile[d].append(mi_matrix[i, j])
    return {d: float(np.mean(vals)) for d, vals in profile.items()}

# ------------------------------------------------------------
# Hamming Weight Correlation
# ------------------------------------------------------------
def hamming_weight(bitstring):
    return bitstring.count("1")

def hamming_distance_correlation(probs):
    bitstrings = list(probs.keys())
    weights = [hamming_weight(b) for b in bitstrings]
    values = [probs[b] for b in bitstrings]

    spearman = spearmanr(values, weights)
    kendall = kendalltau(values, weights)

    return {
        "spearman_corr": float(spearman.correlation),
        "kendall_tau": float(kendall.correlation)
    }

# ------------------------------------------------------------
# Main Analysis
# ------------------------------------------------------------
def analyze_highbond(json_path):
    data = load_json(json_path)

    print("\n=== HIGH-BOND ANALYSIS REPORT ===")
    print(f"File: {json_path}")
    print("----------------------------------")

    # Extract probabilities
    probs, shots = extract_probabilities(data)
    print(f"Total shots: {shots}")
    print(f"Unique bitstrings: {len(probs)}")

    # Probability clustering
    clusters, quantiles = cluster_probabilities(probs)
    stats = cluster_stats(probs, clusters)

    print("\n--- Probability Clusters (Quantile-based) ---")
    print("Quantiles:", quantiles)
    for cid, info in stats.items():
        print(f"\nCluster {cid}:")
        print(f"  Count: {info['count']}")
        print(f"  Min:   {info['min']:.6f}")
        print(f"  Max:   {info['max']:.6f}")
        print(f"  Mean:  {info['mean']:.6f}")
        print(f"  Entropy: {info['entropy']:.6f}")
        print("  Bitstrings:", clusters[cid])

    # MI matrix: use stored MI if present, otherwise compute
    if "mutual_information_matrix" in data:
        mi_matrix = np.array(data["mutual_information_matrix"])
        print("\nUsing MI matrix from JSON.")
    else:
        print("\nComputing MI matrix from counts...")
        mi_matrix = compute_mutual_information(probs)

    # MI spectral analysis
    mi_spec = mi_spectral_analysis(mi_matrix)
    print("\n--- MI Spectral Analysis ---")
    print("Eigenvalues:", mi_spec["eigenvalues"])
    print("Spectral Radius:", mi_spec["spectral_radius"])
    print("Participation Ratio:", mi_spec["participation_ratio"])

    # MI vs distance
    mi_dist = mi_vs_distance(mi_matrix)
    print("\n--- MI vs Distance ---")
    for d, val in mi_dist.items():
        print(f"Distance {d}: MI = {val:.6f}")

    # Hamming distance correlation
    ham_corr = hamming_distance_correlation(probs)
    print("\n--- Hamming Weight Correlation ---")
    print("Spearman:", ham_corr["spearman_corr"])
    print("Kendall Tau:", ham_corr["kendall_tau"])

    print("\n=== END OF REPORT ===\n")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis_highbond.py <json_file>")
        sys.exit(1)

    analyze_highbond(sys.argv[1])
