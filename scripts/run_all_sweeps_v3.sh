#!/bin/sh
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
$PY scripts/sweep_grades.py --base h10 --mode functional --tag swF2c --style-ref h10 --style-w 2.0 --grades 2 3 4 --gens 40 > results_swF2c.log 2>&1
$PY scripts/sweep_grades.py --base h10 --mode functional --tag swF1c --grades 2 3 4 --gens 40 > results_swF1c.log 2>&1
$PY scripts/sweep_grades.py --base h10 --mode mechanical --tag swM1c --grades 2 3 4 --gens 40 > results_swM1c.log 2>&1
echo ALLDONE > results_sweeps_c_done.txt
