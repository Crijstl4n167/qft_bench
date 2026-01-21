systemd-run --scope \
  bash -lc '
    source ~/.venv_q/bin/activate
    exec python python_bench.py -n 32 -d 0
  ' | tee -a python_bench_results.txt
  
