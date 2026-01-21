#!/usr/bin/env python3
import os
import time
import threading
import tracemalloc
import psutil
import cpuinfo
import numpy as np

import pennylane as qml

process = psutil.Process(os.getpid())
peak_rss = 0
running = True


def poll_memory(interval=0.01):
    """Poll RSS memory (resident set size) in a background thread."""
    global peak_rss
    while running:
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        time.sleep(interval)


def random_complex_unit_vector(n: int) -> np.ndarray:
    """Random normalized complex vector of length n."""
    real = np.random.normal(size=n)
    imag = np.random.normal(size=n)
    v = real + 1j * imag
    v /= np.linalg.norm(v)
    return v.astype(np.complex128)


def samples_to_counts(samples: np.ndarray) -> dict:
    """
    Convert PennyLane samples to a Qiskit-like counts dict:
    keys are bitstrings, values are counts.

    samples shape: (shots, n_wires) with entries 0/1
    """
    # Convert each row to a bitstring, e.g. [1 0 1] -> "101"
    bitstrings = ["".join(map(str, row.tolist())) for row in samples]
    unique, counts = np.unique(bitstrings, return_counts=True)
    return {k: int(v) for k, v in zip(unique, counts)}


# -------------------------
# Configuration
# -------------------------
min_qubits = 24
max_qubits = 28 + 1
shots = 1024

# Choose device:
# - "default.qubit": pure Python, baseline
# - "lightning.qubit": faster CPU backend (if available in your install)
device_name = "default.qubit"  # change to "lightning.qubit" if desired
#device_name = "lightning.gpu"
#device_name = "lightning.qubit"


time_init = np.zeros(max_qubits)
time_composition = np.zeros(max_qubits)
time_compile = np.zeros(max_qubits)  # Pennylane "compile" step
time_sim = np.zeros(max_qubits)
time_total = np.zeros(max_qubits)

max_mem = np.zeros(max_qubits)       # RSS peak via psutil polling
max_malloc = np.zeros(max_qubits)    # peak Python allocations via tracemalloc

# Print CPU info (similar to your script)
info = cpuinfo.get_cpu_info()
print(f"CPU Marke/Name: {info.get('brand_raw', 'unknown')}")
print(f"Anzahl Kerne: {info.get('count', 'unknown')}")
print(f"Architektur: {info.get('arch', 'unknown')}")
flags = info.get("flags", [])
print(f"Flags: {flags}")

print(f"PennyLane device: {device_name}, shots={shots}")

for n in range(min_qubits, max_qubits):
    num_qubits = n
    wires = list(range(num_qubits))

    # tracemalloc for Python-level allocation tracking
    tracemalloc.start()

    # --- start RSS polling ---
    running = True
    peak_rss = 0
    thread = threading.Thread(target=poll_memory, daemon=True)
    thread.start()

    # --- init random statevector ---
    print(f"Prepare statevector n={n}")
    start_init = time.perf_counter()
    psi0 = random_complex_unit_vector(2**num_qubits)
    end_init = time.perf_counter()
    init_time = end_init - start_init

    # --- build circuit / QNode ---
    print("Building circuit...")
    start0 = time.perf_counter()

    dev = qml.device(device_name, wires=num_qubits, shots=shots)

    def circuit():
        # Equivalent to initial_statevector=psi0 in Qiskit Aer
        qml.StatePrep(psi0, wires=wires)

        # QFT
        qml.QFT(wires=wires)

        # measurement: sample all qubits -> later convert to counts
        return qml.sample(wires=wires)

    # QNode creation roughly corresponds to “composition”
    qnode = qml.QNode(circuit, dev)

    end0 = time.perf_counter()
    composition_time = end0 - start0

    # --- "compile" step (optional but mirrors a transpile-like phase) ---
    # This applies PennyLane's compilation/optimization pipeline.
    # If you want a pure baseline without compile, set do_compile=False.
    do_compile = True

    print("Starting compile...")
    start1 = time.perf_counter()
    if do_compile:
        qnode_compiled = qml.compile(qnode)
    else:
        qnode_compiled = qnode
    end1 = time.perf_counter()
    compile_time = end1 - start1

    # --- run simulation ---
    print("Run simulation...")
    start2 = time.perf_counter()
    samples = qnode_compiled()
    end2 = time.perf_counter()
    sim_time = end2 - start2

    # --- stop polling ---
    running = False
    thread.join()

    # convert samples -> counts (Qiskit-style)
    counts = samples_to_counts(np.asarray(samples))

    # collect metrics
    time_init[n] = init_time
    time_composition[n] = composition_time
    time_compile[n] = compile_time
    time_sim[n] = sim_time
    time_total[n] = composition_time + compile_time + sim_time
    max_mem[n] = peak_rss

    current, peak = tracemalloc.get_traced_memory()
    max_malloc[n] = peak
    tracemalloc.stop()

    # per-n output line (close to your format)
    print(
        f"{n} ... {time_init[n]:.2f}s ... {time_composition[n]:.2f}s ... "
        f"{time_compile[n]:.2f}s ... {time_sim[n]:.2f}s ... {time_total[n]:.2f}s ... "
        f"{max_mem[n]/1024**2:.2f}MB ... {max_malloc[n]/1024**2:.2f}MB"
    )

print("Summary")
print("Qubits ... init time ... composition time ... compile ... simulate time ... total time ... RSS-peak ... malloc-peak")
for n in range(1, max_qubits):
    print(
        f"{n} ... {time_init[n]:.2f}s ... {time_composition[n]:.2f}s ... "
        f"{time_compile[n]:.2f}s ... {time_sim[n]:.2f}s ... {time_total[n]:.2f}s ... "
        f"{max_mem[n]/1024**2:.2f}MB ... {max_malloc[n]/1024**2:.2f}MB"
    )

