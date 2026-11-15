#!/bin/sh
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
$PY scripts/sweep_grades.py --base h7 --mode functional --tag swF2b --style-ref h7 --style-w 2.0 --grades 2 3 4 --gens 40 > results_swF2b.log 2>&1
$PY scripts/sweep_grades.py --base h7 --mode functional --tag swF1b --grades 2 3 4 --gens 40 > results_swF1b.log 2>&1
$PY scripts/sweep_grades.py --base h7 --mode mechanical --tag swM1b --grades 2 3 4 --gens 40 > results_swM1b.log 2>&1
echo ALLDONE > results_sweeps_b_done.txt
