#!/bin/sh
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until grep -q ALLDONE results_extra_sweeps_progress.txt 2>/dev/null; do sleep 30; done
$PY scripts/sensitivity.py --base h11 --tag sensH2 --style-ref h11 --style-w 2.0 > results_sensH2.log 2>&1
$PY scripts/sensitivity.py --base h11 --tag sensH1 > results_sensH1.log 2>&1
echo ALLDONE > results_sensitivity_done.txt
