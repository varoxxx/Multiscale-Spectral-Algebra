# -*- coding: utf-8 -*-
"""
David Mulnix copyright 2026
"""

import json
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ============================================================
# Utility: run circuit and return counts
# ============================================================
def simulate(qc, shots=2048):
    backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    result = job.result()
    return result.get_counts()

# ============================================================
# Input state preparation
# ============================================================
def prepare_input_state(qc, label):
    if label == "000000":
        return qc

    elif label == "010101":
        for q in [0,2,4]:
            qc.x(q)
        return qc

    elif label == "001111":
        for q in [2,3,4,5]:
            qc.x(q)
        return qc

    elif label == "111000":
        for q in [0,1,2]:
            qc.x(q)
        return qc

    elif label == "allplus":
        for q in range(qc.num_qubits):
            qc.h(q)
        return qc

    elif label == "plusplus":
        qc.h(0)
        qc.h(1)
        return qc

    elif label == "0p0p0p":
        for q in [1,3,5]:
            qc.h(q)
        return qc

    else:
        raise ValueError(f"Unknown input state: {label}")

# ============================================================
# Baseline circuits (NO measurement here)
# ============================================================
def build_cz_circuit(N):
    qc = QuantumCircuit(N)
    qc.cz(0, 1)
    return qc

def build_cnot_circuit(N):
    qc = QuantumCircuit(N)
    qc.cx(0, 1)
    return qc

def build_qft_like_circuit(N):
    qc = QuantumCircuit(N)
    for q in range(N - 1):
        qc.cx(q, q + 1)
    return qc

# ============================================================
# Run all baselines for all input states
# ============================================================
def run_all_baselines(N=6):

    input_states = [
        "000000",
        "001111",
        "010101",
        "0p0p0p",
        "111000",
        "allplus",
        "plusplus"
    ]

    circuits = {
        "cz": build_cz_circuit(N),
        "cnot": build_cnot_circuit(N),
        "qft_like": build_qft_like_circuit(N),
    }

    for base_name, base_qc in circuits.items():
        for label in input_states:

            qc = QuantumCircuit(N, N)

            qc = prepare_input_state(qc, label)

            qc.compose(base_qc, inplace=True)

            qc.measure(range(N), range(N))

            counts = simulate(qc)

            filename = f"{base_name}_{label}.json"
            with open(filename, "w") as f:
                json.dump({"counts": counts}, f, indent=2)

            print(f"Saved {filename}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_all_baselines(N=6)
