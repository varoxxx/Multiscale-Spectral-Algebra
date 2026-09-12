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
def prepare_input_state(qc, input_state):
    """
    Input states are always prepared on the *same* N-qubit register.
    We use:
      - 'plusplus' : H on qubits 0 and 1 → |++⟩ ⊗ |0...0⟩
      - 'plus0'    : H on qubit 0       → |+0⟩ ⊗ |0...0⟩
      - '0plus'    : H on qubit 1       → |0+⟩ ⊗ |0...0⟩
      - 'allplus'  : H on all N qubits  → |++++++⟩
    """
    if input_state == "plusplus":      # |++⟩ on qubits 0 and 1
        qc.h(0)
        qc.h(1)
    elif input_state == "plus0":       # |+0⟩
        qc.h(0)
    elif input_state == "0plus":       # |0+⟩
        qc.h(1)
    elif input_state == "allplus":     # |++++++⟩
        for q in range(qc.num_qubits):
            qc.h(q)
    else:
        raise ValueError(f"Unknown input state: {input_state}")
    return qc

# ============================================================
# Baseline circuits (NO measurement here)
# ============================================================
def build_cz_circuit(N):
    """
    Simple CZ baseline: entangle qubits 0 and 1, others idle.
    """
    qc = QuantumCircuit(N)
    qc.cz(0, 1)
    return qc

def build_cnot_circuit(N):
    """
    Simple CNOT baseline: entangle qubits 0 and 1, others idle.
    """
    qc = QuantumCircuit(N)
    qc.cx(0, 1)
    return qc

def build_qft_like_circuit(N):
    """
    'QFT-like' baseline: a non-trivial multi-qubit entangling pattern
    that does NOT undo the prepared |+⟩ states.

    Here we use a simple entangling chain:
        CX(0→1), CX(1→2), ..., CX(N-2→N-1)

    This is not a full QFT, but it is:
      - genuinely multi-qubit,
      - non-basic compared to single CZ/CNOT,
      - sensitive to the prepared superposition structure.
    """
    qc = QuantumCircuit(N)
    for q in range(N - 1):
        qc.cx(q, q + 1)
    return qc

# ============================================================
# Run all baselines for all input states
# ============================================================
def run_all_baselines(N=6):
    input_states = ["plusplus", "plus0", "0plus", "allplus"]

    circuits = {
        "cz": build_cz_circuit(N),
        "cnot": build_cnot_circuit(N),
        "qft_like": build_qft_like_circuit(N),
    }

    for name, base_qc in circuits.items():
        for state in input_states:
            # Fresh N-qubit circuit with N classical bits
            qc = QuantumCircuit(N, N)

            # Prepare input state on the same N-qubit register
            qc = prepare_input_state(qc, state)

            # Compose the baseline unitary (acts on the same N-qubit register)
            qc.compose(base_qc, inplace=True)

            # Measure qubits 0..N-1 → classical bits 0..N-1
            qc.measure(range(N), range(N))

            # Simulate
            counts = simulate(qc)

            # Save JSON
            filename = f"{name}_{state}.json"
            with open(filename, "w") as f:
                json.dump({"counts": counts}, f, indent=2)

            print(f"Saved {filename}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_all_baselines(N=6)
