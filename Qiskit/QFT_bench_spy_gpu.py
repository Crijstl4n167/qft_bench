
# UPDATED QFT_bench_spy_gpu.py
# Includes RAM + VRAM peak observer via psutil + NVML

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFTGate
import numpy as np
import time, threading, psutil, os, tracemalloc, cpuinfo

# ---------- NVML (VRAM) ----------
from pynvml import *
nvmlInit()
GPU_INDEX = int(os.environ.get("GPU_INDEX", 0))
gpu_handle = nvmlDeviceGetHandleByIndex(GPU_INDEX)

process = psutil.Process(os.getpid())

peak_rss  = 0
peak_vram = 0
running   = True

def poll_memory(interval=0.01):
    global peak_rss
    while running:
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        time.sleep(interval)

def poll_vram(interval=0.01):
    global peak_vram
    while running:
        mem = nvmlDeviceGetMemoryInfo(gpu_handle)
        peak_vram = max(peak_vram, mem.used)
        time.sleep(interval)

def random_complex_unit_vector(n):
    v = np.random.normal(size=n) + 1j*np.random.normal(size=n)
    return v / np.linalg.norm(v)

def print_aer_device(backend):
    print(f"Aer method={backend.options.method}, device={backend.options.device}, available={backend.available_devices()}")

min_qubits = 26
max_qubits = 33 + 1

time_init = np.zeros(max_qubits)
time_comp = np.zeros(max_qubits)
time_trans = np.zeros(max_qubits)
time_sim  = np.zeros(max_qubits)
time_total = np.zeros(max_qubits)

max_mem   = np.zeros(max_qubits)
max_vram  = np.zeros(max_qubits)
max_malloc = np.zeros(max_qubits)

backend = AerSimulator(method="statevector", device="GPU")

info = cpuinfo.get_cpu_info()
print(f"CPU: {info['brand_raw']} | Cores: {info['count']} | Arch: {info['arch']}")
print_aer_device(backend)

for n in range(min_qubits, max_qubits):

    tracemalloc.start()

    peak_rss = peak_vram = 0
    running = True

    t_ram  = threading.Thread(target=poll_memory, daemon=True)
    t_vram = threading.Thread(target=poll_vram, daemon=True)
    t_ram.start()
    t_vram.start()

    start = time.perf_counter()
    psi0 = random_complex_unit_vector(2**n).astype(np.complex128)
    time_init[n] = time.perf_counter() - start

    start = time.perf_counter()
    qc = QuantumCircuit(n, n)
    qc.append(QFTGate(n), range(n))
    qc.measure(range(n), range(n))
    time_comp[n] = time.perf_counter() - start

    start = time.perf_counter()
    circ = transpile(qc, backend)
    time_trans[n] = time.perf_counter() - start

    start = time.perf_counter()
    backend.run(circ, shots=1024, initial_statevector=psi0).result()
    time_sim[n] = time.perf_counter() - start

    running = False
    t_ram.join()
    t_vram.join()

    time_total[n] = time_comp[n] + time_trans[n] + time_sim[n]
    max_mem[n] = peak_rss
    max_vram[n] = peak_vram

    _, peak = tracemalloc.get_traced_memory()
    max_malloc[n] = peak
    tracemalloc.stop()

    print(f"{n} ... {time_init[n]:.2f}s ... {time_comp[n]:.2f}s ... {time_trans[n]:.2f}s ... "
          f"{time_sim[n]:.2f}s ... {time_total[n]:.2f}s ... "
          f"{max_mem[n]/1024**2:.1f}MB RAM ... {max_vram[n]/1024**2:.1f}MB VRAM ... "
          f"{max_malloc[n]/1024**2:.1f}MB malloc")

print("\nSummary")
for n in range(1, max_qubits):
    print(f"{n} ... {time_total[n]:.2f}s ... {max_mem[n]/1024**2:.1f}MB RAM ... "
          f"{max_vram[n]/1024**2:.1f}MB VRAM")

nvmlShutdown()
