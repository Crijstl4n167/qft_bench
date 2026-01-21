from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate
import numpy as np
import time
import threading
import psutil
import os
import tracemalloc
import cpuinfo

# -----------------------------
# Memory + VRAM observer
# -----------------------------
process = psutil.Process(os.getpid())
peak_rss = 0
peak_vram = 0
running = True

# NVML (VRAM) optional
_nvml_ok = False
_nvml_handle = None
_nvml_total_vram = None

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetComputeRunningProcesses,
        nvmlDeviceGetGraphicsRunningProcesses,
        NVMLError,
    )

    nvmlInit()
    _gpu_index = int(os.getenv("GPU_INDEX", "0"))
    _nvml_handle = nvmlDeviceGetHandleByIndex(_gpu_index)
    _nvml_total_vram = nvmlDeviceGetMemoryInfo(_nvml_handle).total
    _nvml_ok = True
except Exception:
    # VRAM tracking stays disabled if NVML isn't available.
    _nvml_ok = False


def _get_process_vram_bytes(handle, pid: int) -> int:
    """Return VRAM used by *this process* (bytes), if supported; else 0.

    Notes:
    - This uses NVML per-process accounting.
    - Depending on driver/settings, usedGpuMemory can be "NVML_VALUE_NOT_AVAILABLE".
    """
    used = 0
    try:
        # Compute processes
        try:
            for p in nvmlDeviceGetComputeRunningProcesses(handle):
                if getattr(p, "pid", None) == pid and getattr(p, "usedGpuMemory", None) is not None:
                    if isinstance(p.usedGpuMemory, int) and p.usedGpuMemory > 0:
                        used += p.usedGpuMemory
        except NVMLError:
            pass

        # Graphics processes
        try:
            for p in nvmlDeviceGetGraphicsRunningProcesses(handle):
                if getattr(p, "pid", None) == pid and getattr(p, "usedGpuMemory", None) is not None:
                    if isinstance(p.usedGpuMemory, int) and p.usedGpuMemory > 0:
                        used += p.usedGpuMemory
        except NVMLError:
            pass

    except Exception:
        return 0

    return used


def poll_resources(interval: float = 0.01):
    global peak_rss, peak_vram
    pid = os.getpid()

    while running:
        # RAM (RSS)
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)

        # VRAM (per-process, if NVML is available)
        if _nvml_ok and _nvml_handle is not None:
            vram_used = _get_process_vram_bytes(_nvml_handle, pid)
            # Fallback: If per-process accounting yields 0 (unsupported), use total device used.
            if vram_used == 0:
                try:
                    vram_used = nvmlDeviceGetMemoryInfo(_nvml_handle).used
                except Exception:
                    vram_used = 0
            peak_vram = max(peak_vram, vram_used)

        time.sleep(interval)


def random_complex_unit_vector(n):
    # Zufälliger komplexer Vektor
    real = np.random.normal(size=n)
    imag = np.random.normal(size=n)
    v = real + 1j * imag

    # Normieren auf Länge 1
    v /= np.linalg.norm(v)
    return v


def print_aer_device(backend):
    print(
        f"Aer method={backend.options.method}, "
        f"device={backend.options.device}, "
        f"available={backend.available_devices()}"
    )


min_qubits = 1
max_qubits = 28 + 1

time_init = np.zeros(max_qubits)
time_composition = np.zeros(max_qubits)
time_transpile = np.zeros(max_qubits)
time_sim = np.zeros(max_qubits)
time_total = np.zeros(max_qubits)

max_mem = np.zeros(max_qubits)       # RSS peak (bytes)
max_malloc = np.zeros(max_qubits)    # tracemalloc peak (bytes)
max_vram = np.zeros(max_qubits)      # VRAM peak (bytes)

backend = AerSimulator(method="statevector")

# print CPU info
info = cpuinfo.get_cpu_info()
print(f"CPU Marke/Name: {info['brand_raw']}")
print(f"Anzahl Kerne: {info['count']}")
print(f"Architektur: {info['arch']}")
print(f"Flags: {info['flags']}")  # Viele Details zu CPU-Funktionen

print_aer_device(backend)

if _nvml_ok:
    total_mib = _nvml_total_vram / 1024**2 if _nvml_total_vram else 0
    gpu_index = int(os.getenv("GPU_INDEX", "0"))
    print(f"NVML: enabled (GPU_INDEX={gpu_index}, total VRAM={total_mib:.0f} MiB)")
else:
    print("NVML: disabled (install: pip install nvidia-ml-py3; requires NVIDIA driver + nvidia-smi)")


for n in range(min_qubits, max_qubits):
    num_qubits = n

    tracemalloc.start()

    # --- Polling starten ---
    running = True
    peak_rss = 0
    peak_vram = 0
    thread = threading.Thread(target=poll_resources, daemon=True)
    thread.start()

    print(f"Prepare statevector n={n}")
    start_init = time.perf_counter()
    psi0 = random_complex_unit_vector(2**num_qubits).astype(np.complex128)
    end_init = time.perf_counter()
    init_time = end_init - start_init

    ### QFT Circuit ###
    print("Building circuit...")
    start0 = time.perf_counter()
    circuit = QuantumCircuit(n, n)
    circuit.append(QFTGate(n), range(n))
    circuit.measure(range(n), range(n))
    end0 = time.perf_counter()
    composition_time = end0 - start0

    print("Starting transpile...")
    start1 = time.perf_counter()
    circ = transpile(circuit, backend)  # Rewrites circuit to match backend basis gates & coupling map
    end1 = time.perf_counter()
    transpile_time = end1 - start1

    print("Run simulation...")
    start2 = time.perf_counter()
    result = backend.run(circ, shots=1024, initial_statevector=psi0).result()
    end2 = time.perf_counter()
    sim_time = end2 - start2

    # --- Polling stoppen ---
    running = False
    thread.join()

    counts = result.get_counts(circ)
    time_init[n] = init_time
    time_composition[n] = composition_time
    time_transpile[n] = transpile_time
    time_sim[n] = sim_time
    time_total[n] = composition_time + transpile_time + sim_time

    max_mem[n] = peak_rss
    max_vram[n] = peak_vram

    current, peak = tracemalloc.get_traced_memory()
    max_malloc[n] = peak

    print(
        f"{n} ... {time_init[n]:.2f}s ... {time_composition[n]:.2f}s ... {time_transpile[n]:.2f}s ... "
        f"{time_sim[n]:.2f}s ... {time_total[n]:.2f}s ... {max_mem[n]/1024**2:.1f}MB ... "
        f"{max_vram[n]/1024**2:.1f}MB_VRAM ... {max_malloc[n]/1024**2:.1f}MB_malloc"
    )


print("Summary")
print(
    "Qubits  ... init time ... composition time ... transpile ... simulation time ... total time "
    "... RAM_peak_MB ... VRAM_peak_MB ... Py_malloc_peak_MB"
)
for n in range(1, max_qubits):
    print(
        f"{n} ... {time_init[n]:.2f}s ... {time_composition[n]:.2f}s ... {time_transpile[n]:.2f}s ... "
        f"{time_sim[n]:.2f}s ... {time_total[n]:.2f}s ... {max_mem[n]/1024**2:.1f}MB ... "
        f"{max_vram[n]/1024**2:.1f}MB ... {max_malloc[n]/1024**2:.1f}MB"
    )

# Cleanup NVML
try:
    if _nvml_ok:
        nvmlShutdown()
except Exception:
    pass
