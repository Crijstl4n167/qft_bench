systemd-run --scope \
  bash -lc '
    source ~/.venv_q/bin/activate
    exec python -u QFT_bench_spy_opti.py
  ' | tee -a QFT_bench_spy_opti.log

