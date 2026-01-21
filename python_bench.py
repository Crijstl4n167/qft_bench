import time
import gc
import csv
import argparse
import tracemalloc
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from contextlib import contextmanager
from python_sim import (
    QFT, SQFT, IQFT,
    QFTS, SQFTS, IQFTS, 
    QFTN, SQFTN, IQFTN, 
)

pd.set_option(
    'display.float_format', 
    lambda x: f'{x:09.6f}'
)

def measure_peak(func, args=(), kwargs=None, conn=None):
    kwargs = kwargs or {}

    gc.collect()
    tracemalloc.start()

    func(*args, **kwargs)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    conn.send(peak)
    conn.close()

def run_with_peak(func, *args, **kwargs):
    parent, child = mp.Pipe()
    p = mp.Process(target=measure_peak, args=(func, args, kwargs, child))
    p.start()
    peak = parent.recv()
    p.join()
    return peak

def format_bytes(x):
    if pd.isna(x):
        return ""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(x) < 1024:
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} EB"

def create_random_state(dim: int) -> np.ndarray:
    real = np.random.normal(size=2**dim)
    imag = np.random.normal(size=2**dim)
    v = real + 1j * imag
    v /= np.linalg.norm(v)
    return v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-qbits", 
        help="maximal number of qbits to use", 
        required=True
    )
    parser.add_argument("-d", "--num-dense",
        help="maximal number of qbits for dense runs"
    )
    args = parser.parse_args()

    max_qbits = int(args.num_qbits)
    if (args.num_dense):
        max_dense = int(args.num_dense)
    else:
        max_dense = max_qbits
    dims = list(range(1, max_qbits+1))
    dense_methods = {
        'QFT': QFT, 
        'SQFT': SQFT, 
        # 'IQFT': IQFT
    }
    numba_methods = {
        'QFTN': QFTN, 
        'SQFTN': SQFTN, 
        # 'IQFTN': IQFTN
    }
    sparse_methods = {
        'QFTS': QFTS,
        'SQFTS': SQFTS,
        # 'IQFTS': IQFTS
    }
    methods = {
        **dense_methods,
        # **numba_methods,
        **sparse_methods
    }

    times = pd.DataFrame(
        columns=methods.keys(), 
        index=dims
    )

    mems = pd.DataFrame(
        columns=methods.keys(),
        index=dims
    )

    mems.style.format(format_bytes, subset=mems.select_dtypes("number").columns)

    with open('python_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['DIM'] + list(methods.keys()) + list(methods.keys()))

    try:
        for dim in tqdm(dims):
            state = create_random_state(dim)

            temp_times = {}
            temp_mems = {}
            method_iter = tqdm(methods.items(), leave=False)
            for key, method in method_iter:
                start = time.process_time()
                method(state.copy())
                end = time.process_time()
                temp_times[key] = end - start
                temp_mems[key] = run_with_peak(method, state.copy())

            times.loc[dim] = temp_times
            mems.loc[dim] = temp_mems
            with open('python_results.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dim] + list(temp_times.values()) + list(temp_mems.values()))

            if dim == max_dense:
                methods = sparse_methods
    except KeyboardInterrupt:
        pass

    mems_pretty_print = mems.map(format_bytes)


    print()
    print('--------------')
    print('---  Time  ---')
    print('--------------')
    print(times)

    print()
    print('--------------')
    print('--- Memory ---')
    print('--------------')
    print(mems_pretty_print)

# vim:ts=4 sw=4 et:
