systemd-run --scope \
  bash -lc '
    source ~/.venv_q/bin/activate
    exec python qft_test.py
  ' | tee -a qft_test_results.txt

