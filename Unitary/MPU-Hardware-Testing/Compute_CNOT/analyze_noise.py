"""
David Mulnix copyright 2026
"""


import json
import glob
import statistics

def to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None

def summarize_noise(fname):
    print(f"\n=== Noise Summary for {fname} ===")

    with open(fname, "r") as f:
        data = json.load(f)

    # Extract raw fields
    t1_raw = data.get("t1", [])
    t2_raw = data.get("t2", [])
    readout_raw = data.get("readout", [])
    cx_raw = data.get("cx_gate_error", None)

    # Convert to floats
    t1_sec = [to_float(x) for x in t1_raw if to_float(x) is not None]
    t2_sec = [to_float(x) for x in t2_raw if to_float(x) is not None]
    readout = [to_float(x) for x in readout_raw if to_float(x) is not None]
    cx = to_float(cx_raw)

    # Convert seconds → microseconds
    t1 = [x * 1e6 for x in t1_sec]
    t2 = [x * 1e6 for x in t2_sec]

    def summarize(values, label):
        if not values:
            print(f"{label}: no numeric data")
        else:
            print(f"{label}: min={min(values):.2f}, max={max(values):.2f}, mean={statistics.mean(values):.2f}")

    print(f"Backend: {data.get('backend')}")
    print(f"Circuit depth: {data.get('depth')}")
    print(f"Gate counts: {data.get('gate_counts')}")

    print("\n--- Noise Metrics ---")
    summarize(t1, "T1 (µs)")
    summarize(t2, "T2 (µs)")
    summarize(readout, "Readout error")

    print(f"CX gate error: {cx if cx is not None else 'no numeric data'}")

    print("\n=== End of Summary ===\n")


if __name__ == "__main__":
    files = glob.glob("ibm_run_multilayer_N2_*.json")
    if not files:
        print("No JSON files found.")
        exit(1)

    for fname in files:
        summarize_noise(fname)
