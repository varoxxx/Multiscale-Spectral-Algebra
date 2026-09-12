
"""
David Mulnix copyright 2026

Multi‑Layer MPU Test: Purpose, Geometric Math, and How to Read the Results

This script implements the **multi‑layer MPU use case** using the geometric
extraction–projection pipeline. In the MPU paper, a “multi‑layer MPU” is an
operator built as a *product of MPU layers*, each layer itself being a
nearest‑neighbor two‑qubit MPU. Algebraically, the paper constructs such
operators using a sequence of bulk tensors and merging steps. Here, instead of
using their algebraic rules, we begin with the full global operator and recover
its MPU structure directly through geometric tensor methods.

Geometric viewpoint
-------------------
The geometric MPU method treats any global unitary U ∈ U(2^N) as a rank‑2N
tensor, interleaves its input/output indices, and performs a site‑wise SVD
factorization. Each SVD produces a local tensor A^{(k)} whose bond dimensions
reflect the entanglement structure at that site. A multi‑layer MPU operator
U_multi = U₂ U₁ is therefore “seen” geometrically as a single global unitary
whose singular‑value structure encodes the combined entanglement of both layers.

The geometric pipeline consists of:
  • Interleaving U into U_{i1 j1 i2 j2 … iN jN}
  • Site‑wise SVD extraction:
        A^{(k)}_{α,i,j,β} = U^{(k)}_{α i j, β} · Σ^{(k)}_{β}
  • Bond‑dimension propagation: bond_left → χ_eff
  • Tensor contraction of A‑tensors along bond indices
  • Polar re‑unitarization (nearest unitary)
  • Global phase alignment
  • Validation via operator fidelity, action‑on‑states, Schmidt ranks,
    and reconstruction checks


What this test does
-------------------
1. Builds a **multi‑layer MPU operator**:
       U_multi = U₂ @ U₁
   where each U₁ and U₂ is a random high‑bond MPU layer built from nearest‑neighbor
   two‑qubit gates.

2. Applies the geometric MPU projection to obtain U_mpu.

3. Validates correctness through:
   • Frobenius(U_multi − U_mpu)
   • Operator fidelity(U_mpu, U_multi)
   • A‑tensor shapes (showing the multi‑layer bond profile)
   • Action‑on‑states differences (≈ 10⁻¹⁵)
   • Schmidt ranks across bipartitions
   • Reconstruction fidelity
   • Axis‑ordering reconstruction check

How to read the results
-----------------------
• Multi‑layer Frobenius = 0.0  
  → The projected MPU operator matches the target multi‑layer operator up to
    numerical precision.

• Operator fidelity = 1.0  
  → Exact equality up to global phase.

• Action‑on‑states ≈ 3.5×10⁻¹⁵  
  → For a 64×64 unitary, this is excellent numerical agreement.

• Schmidt ranks = {4, 16, 64, 16, 4}  
  → This is the expected multi‑layer high‑bond profile: large bond in the bulk,
    smaller at the boundaries.

• Reconstruction fidelity = 1.0  
  → The A‑tensor chain reconstructs the operator exactly (up to phase).

Important note about reconstruction Frobenius
---------------------------------------------
When reconstructing U_rec from A‑tensors, Frobenius behaves differently:

• In the multi‑layer test, U_rec and U_target differ by a *global phase*.  
• Frobenius norm is sensitive to global phase, even though fidelity is not.  
• For a 64×64 unitary, a pure global phase can produce Frobenius ≈ 16.  
• Fidelity = 1.0 proves the reconstruction is exact (up to phase).  
• Therefore:
      – Frobenius ≈ 16 is NOT an error in this context.
      – Fidelity is the correct metric for reconstruction.
      – In other parts of the test (projection), Frobenius = 0 IS expected.

Summary
-------
This test demonstrates that the geometric MPU extraction–projection pipeline
successfully reconstructs a multi‑layer MPU operator—one of the key MPU use
cases—with operator fidelity 1.0 and full functional correctness. These results
show that multi‑layer MPU circuits can be represented using geometric tensor
methods, without relying on the algebraic MPU machinery.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Operator
import cupy as cp
from module_fidelity_objective import FidelityObjective
import json
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# ============================================================
# MPU extraction sweep
# ============================================================

def mpu_extract_all_sites(U, n_qubits, chi, debug=False):
    d = 2
    N = n_qubits

    U_tensor = U.reshape((d,)*N + (d,)*N)
    interleave = []
    for k in range(N):
        interleave.append(k)
        interleave.append(N+k)
    U_inter = U_tensor.transpose(interleave)

    local_dim = d**2
    rest_dim = d**(2*(N-1))
    M = U_inter.reshape(local_dim, rest_dim)

    A_list = []
    bond_left_dim = 1

    for site in range(N):
        Umat, Svec, Vh = cp.linalg.svd(M, full_matrices=False)
        chi_eff = min(chi, Umat.shape[1])

        Utr = Umat[:, :chi_eff]
        Str = Svec[:chi_eff]
        Vtr = Vh[:chi_eff, :]

        A_site = cp.zeros((bond_left_dim, d, d, chi_eff), dtype=cp.complex128)
        for alpha in range(bond_left_dim):
            for i in range(d):
                for j in range(d):
                    idx = alpha*d**2 + i*d + j
                    for beta in range(chi_eff):
                        A_site[alpha, i, j, beta] = Utr[idx, beta] * Str[beta]

        A_list.append(A_site)

        if site < N-1:
            rest_dim_next = d**(2*(N-site-2))
            M = Vtr.reshape(chi_eff * d**2, rest_dim_next)
            bond_left_dim = chi_eff

    if debug:
        print("\n--- MPU Multi-Site Extraction ---")
        for k, A in enumerate(A_list):
            print(f"A[{k}] shape: {A.shape}")

    return A_list


# ============================================================
# MPU projection (SVD-based)
# ============================================================

def mpu_projection(U, n_qubits, chi, debug=False):
    N = n_qubits
    dim = 2**N

    A_list = mpu_extract_all_sites(U, N, chi, debug=debug)

    T = A_list[0]
    for k in range(1, N):
        Ak = A_list[k]

        m = min(T.shape[-1], Ak.shape[0])
        T = T[..., :m]
        Ak = Ak[:m, :, :, :]

        T = cp.tensordot(T, Ak, axes=([-1], [0]))

    l = cp.ones((T.shape[0],), dtype=cp.complex128)
    r = cp.ones((T.shape[-1],), dtype=cp.complex128)

    T = cp.tensordot(l.conj(), T, axes=([0], [0]))
    T = cp.tensordot(T, r, axes=([-1], [0]))

    axes = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U_mpu = T.transpose(axes).reshape(dim, dim)

    Ux, Sx, Vxh = cp.linalg.svd(U_mpu, full_matrices=False)
    return Ux @ Vxh


def align_global_phase(U_ref, U):
    num = cp.vdot(U.flatten(), U_ref.flatten())
    phase = num / cp.abs(num + 1e-18)
    return phase * U

def validate_action_on_states(U_target, U_mpu, n_qubits, trials=20):
    print("\n--- Action-on-States Validation ---")
    dim = 2**n_qubits

    for t in range(trials):
        psi = cp.random.randn(dim) + 1j * cp.random.randn(dim)
        psi /= cp.linalg.norm(psi)

        out_target = U_target @ psi
        out_mpu = U_mpu @ psi

        diff = cp.linalg.norm(out_target - out_mpu)
        print(f"Trial {t}: ||U_target ψ - U_mpu ψ|| = {float(diff):.6e}")

def validate_schmidt_ranks(U, n_qubits):
    print("\n--- Schmidt Rank / Bond Dimension Validation ---")

    for cut in range(1, n_qubits):
        left_dim = 2**cut
        right_dim = 2**(n_qubits - cut)

        # Reshape into bipartition
        U_reshaped = U.reshape(left_dim, right_dim, left_dim, right_dim)

        # Permute to (left_in, left_out) × (right_in, right_out)
        U_perm = cp.transpose(U_reshaped, (0, 2, 1, 3))

        # Flatten into matrix for Schmidt rank
        M = U_perm.reshape(left_dim**2, right_dim**2)

        rank = cp.linalg.matrix_rank(M)
        print(f"Cut {cut}: Schmidt rank = {int(rank)}")

def random_two_qubit_unitary(seed=None):
    if seed is not None:
        cp.random.seed(seed)

    # Random complex 4x4
    M = cp.random.randn(4, 4) + 1j * cp.random.randn(4, 4)

    # QR decomposition
    Q, R = cp.linalg.qr(M)

    # Make unitary with det=1 (optional, but fine)
    det = cp.linalg.det(Q)
    Q = Q / cp.power(det, 1/4)

    return Q

def build_random_mpu_layer(N=6, depth=2, seed=None):
    I2 = cp.eye(2, dtype=cp.complex128)

    def embed_two_qubit_gate(G, q, N):
        U_full = cp.eye(1, dtype=cp.complex128)
        k = 0
        while k < N:
            if k == q:
                U_full = cp.kron(U_full, G)
                k += 2
            else:
                U_full = cp.kron(U_full, I2)
                k += 1
        return U_full

    if seed is not None:
        cp.random.seed(seed)

    dim = 2**N
    U = cp.eye(dim, dtype=cp.complex128)

    # depth layers of nearest-neighbor random 2-qubit gates
    for layer in range(depth):
        for q in range(N - 1):
            G = random_two_qubit_unitary()
            U_layer = embed_two_qubit_gate(G, q, N)
            U = U_layer @ U

    return U

def reconstruct_from_A(A_list):
    """
    Reconstruct full unitary from A-tensors.
    A_list[k] shape: (bond_left, 2, 2, bond_right)
    """
    N = len(A_list)

    # Start with first tensor
    T = A_list[0]

    # Contract chain left → right
    for k in range(1, N):
        Ak = A_list[k]

        # Match bond dimensions
        m = min(T.shape[-1], Ak.shape[0])
        T = T[..., :m]
        Ak = Ak[:m, :, :, :]

        # Contract right bond of T with left bond of Ak
        T = cp.tensordot(T, Ak, axes=([-1], [0]))

    # Contract left boundary (bond dimension 1)
    l = cp.ones((T.shape[0],), dtype=cp.complex128)
    T = cp.tensordot(l.conj(), T, axes=([0], [0]))

    # Contract right boundary (bond dimension 1)
    r = cp.ones((T.shape[-1],), dtype=cp.complex128)
    T = cp.tensordot(T, r, axes=([-1], [0]))

    # Reorder physical indices: (in0, out0, in1, out1, ..., inN-1, outN-1)
    axes = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U = T.transpose(axes).reshape(2**N, 2**N)

    return U

def validate_reconstruction(U_target, A_list):
    print("\n--- Reconstruction Consistency Validation ---")
    U_rec = reconstruct_from_A(A_list)

    diff = cp.linalg.norm(U_target - U_rec)
    fid = FidelityObjective.operator_fidelity(U_rec, U_target)

    print(f"Reconstruction Frobenius: {float(diff):.6e}")
    print(f"Reconstruction Fidelity:  {float(fid):.6f}")

    
def debug_reconstruction(U_target, U_mpu, A_list):
    print("\n--- Reconstruction Debug ---")

    U_rec = reconstruct_from_A(A_list)

    diff_target = cp.linalg.norm(U_target - U_rec)
    diff_mpu    = cp.linalg.norm(U_mpu    - U_rec)

    fid_target = FidelityObjective.operator_fidelity(U_rec, U_target)
    fid_mpu    = FidelityObjective.operator_fidelity(U_rec, U_mpu)

    print(f"Frob(U_rec - U_target): {float(diff_target):.6e}")
    print(f"Frob(U_rec - U_mpu):    {float(diff_mpu):.6e}")
    print(f"Fid(U_rec, U_target):   {float(fid_target):.6f}")
    print(f"Fid(U_rec, U_mpu):      {float(fid_mpu):.6f}")

def reconstruct_from_A_noboundary(A_list):
    """
    Reconstruct full unitary from A-tensors without extra boundary vectors.
    A_list[k] shape: (bond_left, 2, 2, bond_right)
    """
    N = len(A_list)

    # Start with first tensor
    T = A_list[0]

    # Contract chain left → right
    for k in range(1, N):
        Ak = A_list[k]

        m = min(T.shape[-1], Ak.shape[0])
        T = T[..., :m]
        Ak = Ak[:m, :, :, :]

        T = cp.tensordot(T, Ak, axes=([-1], [0]))

    # At this point T.shape = (1, 2, 2, ..., 2, 2, 1)
    # Drop trivial bond dims by reshape
    T = T.reshape(*([2] * (2 * N)))

    # Option A: (in0, out0, in1, out1, ..., inN-1, outN-1)
    axes_A = list(range(0, 2*N, 2)) + list(range(1, 2*N, 2))
    U_A = T.transpose(axes_A).reshape(2**N, 2**N)

    return U_A

def debug_reconstruction_axes(U_target, U_mpu, A_list):
    print("\n--- Reconstruction Axis Debug ---")

    U_A = reconstruct_from_A_noboundary(A_list)

    for label, U_rec in [("A", U_A)]:
        diff_target = cp.linalg.norm(U_target - U_rec)
        diff_mpu    = cp.linalg.norm(U_mpu    - U_rec)

        fid_target = FidelityObjective.operator_fidelity(U_rec, U_target)
        fid_mpu    = FidelityObjective.operator_fidelity(U_rec, U_mpu)

        print(f"[{label}] Frob(U_rec - U_target): {float(diff_target):.6e}")
        print(f"[{label}] Frob(U_rec - U_mpu):    {float(diff_mpu):.6e}")
        print(f"[{label}] Fid(U_rec, U_target):   {float(fid_target):.6f}")
        print(f"[{label}] Fid(U_rec, U_mpu):      {float(fid_mpu):.6f}")
        print()

def build_multi_layer_mpu_operator(N=6, chi=16, depth=2, seed=None):
    """
    Build a multi-layer MPU operator as a product of two MPU layers.
    Each layer is a random high-bond MPU built by build_random_mpu_layer.
    """
    # First layer
    U1 = build_random_mpu_layer(N=N, depth=depth, seed=seed)

    # Second layer (different random instance if seed is None or changed)
    U2 = build_random_mpu_layer(N=N, depth=depth, seed=None if seed is None else seed + 1)

    # Multi-layer operator: U_multi = U2 @ U1
    U_multi = U2 @ U1
    return U_multi


def run_multi_layer_mpu_test():
    print("\n--- Multi-Layer MPU Test ---")

    N = 6
    depth = 2
    chi = 64  

    # Build target multi-layer operator
    U_multi = build_multi_layer_mpu_operator(N=N, chi=chi, depth=depth, seed=123)

    # Project onto MPU manifold
    U_mpu = mpu_projection(U_multi, N, chi, debug=True)
    U_mpu = align_global_phase(U_multi, U_mpu)

    # Operator-level metrics
    frob = float(cp.linalg.norm(U_multi - U_mpu))
    fid  = FidelityObjective.operator_fidelity(U_mpu, U_multi)

    print(f"Multi-layer Frobenius:        {frob:.6f}")
    print(f"Multi-layer operator fidelity:{fid:.6f}")

    # Extract A-tensors from the target operator
    A_list = mpu_extract_all_sites(U_multi, N, chi, debug=True)

    # Validations
    validate_action_on_states(U_multi, U_mpu, N)
    validate_schmidt_ranks(U_mpu, N)
    validate_reconstruction(U_multi, A_list)

    # Optional debug reconstruction checks
    debug_reconstruction(U_multi, U_mpu, A_list)
    debug_reconstruction_axes(U_multi, U_mpu, A_list)

def run_qiskit_test(U_mpu, n_qubits):
    print("\n--- Qiskit Compatibility Test ---")

    U_np = np.array(U_mpu.get())
    op = Operator(U_np)

    instr = op.to_instruction()
    qc = QuantumCircuit(n_qubits)
    qc.append(instr, range(n_qubits))

    print("\n--- Qiskit Circuit Synthesis Successful ---")
    print(qc)
    print("\nCircuit depth:", qc.depth())
    print("Number of gates:", qc.size())

    return qc

def run_qiskit_equivalence_test(U_mpu, qc):
    print("\n--- Qiskit Equivalence Test ---")

    U_mpu_np = np.array(U_mpu.get())
    U_qiskit = Operator(qc).data

    frob = np.linalg.norm(U_mpu_np - U_qiskit)
    fid = (np.abs(np.trace(U_mpu_np.conj().T @ U_qiskit))**2) / (U_mpu_np.shape[0]**2)

    print(f"Frobenius ||U_mpu - U_qiskit||_F = {frob:.6e}")
    print(f"Operator fidelity = {fid:.6e}")

    return frob, fid

def run_qiskit_decomposition(U_mpu, n_qubits):
    print("\n--- Qiskit Decomposition Test (CX + U3) ---")

    U_np = np.array(U_mpu.get())
    op = Operator(U_np)

    instr = op.to_instruction()
    qc = QuantumCircuit(n_qubits)
    qc.append(instr, range(n_qubits))

    qc_decomp = transpile(qc, basis_gates=["cx", "u3"], optimization_level=1)

    print(qc_decomp)
    print("\nDecomposed circuit depth:", qc_decomp.depth())
    print("Decomposed circuit gate count:", qc_decomp.size())

    return qc_decomp

def run_qiskit_decomposition_equivalence(U_mpu, qc_decomp):
    print("\n--- Qiskit Decomposition Equivalence Test ---")

    U_mpu_np = np.array(U_mpu.get())
    U_qiskit = Operator(qc_decomp).data

    frob = np.linalg.norm(U_mpu_np - U_qiskit)
    fid = (np.abs(np.trace(U_mpu_np.conj().T @ U_qiskit))**2) / (U_mpu_np.shape[0]**2)

    print(f"Frobenius ||U_mpu - U_qiskit||_F = {frob:.6e}")
    print(f"Operator fidelity = {fid:.6e}")

    return frob, fid



if __name__ == "__main__":


    ###########################################################################
    # JSON-SAFE CONVERSION LAYER (bulletproof)
    ###########################################################################
    def json_safe(obj):
        """Convert Qiskit objects into JSON-safe primitives."""
        import numpy as np

        # Already JSON-safe
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # numpy types
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # Lists / tuples
        if isinstance(obj, (list, tuple)):
            return [json_safe(x) for x in obj]

        # Dicts
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}

        # CouplingMap or EdgeList
        if hasattr(obj, "get_edges"):
            return [list(edge) for edge in obj.get_edges()]

        # Qiskit gate objects
        if hasattr(obj, "name") and hasattr(obj, "parameters"):
            return {
                "name": obj.name,
                "qubits": json_safe(getattr(obj, "qubits", None)),
                "parameters": json_safe(obj.parameters)
            }

        # Qiskit objects with to_dict()
        if hasattr(obj, "to_dict"):
            return json_safe(obj.to_dict())

        # datetime
        if hasattr(obj, "isoformat"):
            return obj.isoformat()

        # Fallback: convert to string
        return str(obj)

    ###########################################################################
    # OFFLINE SIMULATION MODE (NO IBM CLOUD CALLS)
    ###########################################################################
    SIMULATE_ONLY = True   # <-- SET TO False FOR REAL HARDWARE RUN
    
    if SIMULATE_ONLY:
        print("\n=== OFFLINE SIMULATION MODE: Testing JSON serialization ===")
    
        # Fake backend-like object
        class FakeBackend:
            name = "fake_marrakesh"
    
            def __init__(self):
                self._coupling_map = [(0,1),(1,2),(2,3),(3,4),(4,5)]
                self._basis_gates = ["cx", "rz", "sx", "x"]
                self._config = {"n_qubits": 6, "simulated": True}
                self._status = {"state": "simulated", "operational": True}
    
            @property
            def coupling_map(self):
                class EdgeList:
                    def get_edges(self_non):
                        return self._coupling_map
                return EdgeList()
    
            @property
            def basis_gates(self):
                return self._basis_gates
    
            def configuration(self):
                return self._config
    
            def status(self):
                return self._status
    
        backend = FakeBackend()
    
        # Build fake JSON
        fake_json = json_safe({
            "backend": backend.name,
            "coupling_map": backend.coupling_map,
            "basis_gates": backend.basis_gates,
            "config": backend.configuration(),
            "status": backend.status(),
        })
    
        with open("simulation_test_backend.json", "w") as f:
            json.dump(fake_json, f, indent=2)
    
        print("Simulation successful. JSON serialization works.")
        print("Switch SIMULATE_ONLY = False for real hardware run.")
        exit(0)


    ###########################################################################
    # REAL HARDWARE RUN BELOW
    ###########################################################################

    # ============================================================
    # 1) BUILD MULTILAYER MPU OPERATOR (N = 6)
    # ============================================================
    N = 6
    depth = 2
    chi = 64

    print(f"\n=== Multilayer MPU Test: N = {N} ===")

    U_multi = build_multi_layer_mpu_operator(N=N, chi=chi, depth=depth, seed=123)
    U_mpu_multi = mpu_projection(U_multi, N, chi)
    U_mpu_multi = align_global_phase(U_multi, U_mpu_multi)

    print(f"Operator Frobenius ||U_multi - U_mpu||_F = {float(cp.linalg.norm(U_multi - U_mpu_multi)):.6e}")
    print(f"Operator fidelity F(U_mpu, U_multi) = {FidelityObjective.operator_fidelity(U_mpu_multi, U_multi):.6f}")

    qc = run_qiskit_test(U_mpu_multi, N)
    run_qiskit_equivalence_test(U_mpu_multi, qc)

    qc_decomp = run_qiskit_decomposition(U_mpu_multi, N)
    run_qiskit_decomposition_equivalence(U_mpu_multi, qc_decomp)

    qc_decomp.h(0) #Case 2 Test B Input Case 2 — |+0⟩

    qc_meas = qc_decomp.copy()
    qc_meas.measure_all()

    with open("qc_multilayer_N6.qpy", "wb") as f:
        qpy.dump(qc_meas, f)

    print("\nSaved circuit qc_multilayer_N6.qpy")

    # ============================================================
    # 2) HARDWARE SETUP
    # ============================================================
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="X", #API Key
        instance="YOUR_TEST"
    )

    backend = service.backend("ibm_marrakesh")
    sampler = SamplerV2(backend)

    print("\n=== Available Backends ===")
    print(service.backends())

    # ============================================================
    # 3) TRANSPILATION
    # ============================================================
    qc_trans = transpile(
        qc_meas,
        backend=backend,
        optimization_level=1
    )

    print("\n--- Transpiled Circuit ---")
    print(qc_trans)
    print("Transpiled depth:", qc_trans.depth())
    print("Transpiled gate counts:", qc_trans.count_ops())

    # ============================================================
    # 4) RUN ON HARDWARE
    # ============================================================
    job = sampler.run([qc_trans], shots=2048)
    print("Job ID:", job.job_id())

    result = job.result()
    counts = result[0].data.meas.get_counts()

    print("\n--- IBM Hardware Output (Counts) ---")
    print(counts)

    # ============================================================
    # 5) BACKEND CALIBRATION (CRASH-PROOF VERSION)
    # ============================================================
    props = backend.properties()
    
    # SAFE extraction of T1/T2/readout errors
    t1_times = []
    t2_times = []
    readout_errors = []
    
    for q in range(backend.num_qubits):
        # T1
        try:
            t1_times.append(props.t1(q))
        except Exception:
            t1_times.append(None)
    
        # T2
        try:
            t2_times.append(props.t2(q))
        except Exception:
            t2_times.append(None)
    
        # Readout error
        try:
            readout_errors.append(props.readout_error(q))
        except Exception:
            readout_errors.append(None)
    
    # SAFE extraction of gate errors
    gate_errors = {}
    for gate in props.gates:
        try:
            gate_errors[gate.name] = {
                "qubits": gate.qubits,
                "parameters": gate.parameters
            }
        except Exception:
            gate_errors[gate.name] = None
    
    # SAFE CX gate error
    try:
        cx_gate_error = props.gate_error("cx")
    except Exception:
        cx_gate_error = None
    
    # SAFE backend config + status
    try:
        config = backend.configuration()
    except Exception:
        config = None
    
    try:
        status = backend.status()
    except Exception:
        status = None


    # ============================================================
    # 6) SAVE JSON (FULLY JSON-SAFE)
    # ============================================================
    output_json = json_safe({
        "job_id": job.job_id(),
        "backend": backend.name,
        "counts": counts,

        "depth": qc_trans.depth(),
        "gate_counts": qc_trans.count_ops(),
        "coupling_map": backend.coupling_map,
        "basis_gates": backend.basis_gates,

        "gate_errors": gate_errors,
        "t1": t1_times,
        "t2": t2_times,
        "readout": readout_errors,
        "cx_gate_error": cx_gate_error,

        "config": config,
        "status": status
    })

    with open("ibm_run_multilayer_N6.json", "w") as f:
        json.dump(output_json, f, indent=2)

    print("\nSaved Multilayer JSON: ibm_run_multilayer_N6.json")
