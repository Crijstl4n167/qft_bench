#!/bin/bash
source ~/.venv_q/bin/activate
python plot_summary_results6.py summary_results.txt --no-show --save-dir plots6a
python plot_summary_results6lin.py summary_results.txt --no-show --save-dir plots6a_lin
