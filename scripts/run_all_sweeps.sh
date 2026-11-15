#!/bin/sh
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
$PY scripts/sweep_grades.py --base h6 --mode functional --tag swF2 --style-ref h6 --style-w 2.0 --grades 2 3 4 --gens 40 > results_swF2.log 2>&1
$PY scripts/sweep_grades.py --base h6 --mode functional --tag swF1 --grades 2 3 4 --gens 40 > results_swF1.log 2>&1
$PY scripts/sweep_grades.py --base h6 --mode mechanical --tag swM1 --grades 2 3 4 --gens 40 > results_swM1.log 2>&1
echo ALLDONE > results_sweeps_done.txt
