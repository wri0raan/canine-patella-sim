#!/bin/sh
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until [ -f results/swF2d1_summary.csv ]; do sleep 20; done
$PY scripts/sweep_grades.py --base h11 --mode mechanical --tag swM1d1 --grades 2 3 4 --gens 40 --seed 1 > results_swM1d1.log 2>&1
$PY scripts/sweep_grades.py --base h11 --mode functional --tag swF1d1 --grades 2 3 4 --gens 40 --seed 1 > results_swF1d1.log 2>&1
echo "seed 1 done (all three settings)" >> results_final_sweeps_progress.txt
for SEED in 2 3; do
  $PY scripts/sweep_grades.py --base h11 --mode functional --tag swF2d$SEED --style-ref h11 --style-w 2.0 --grades 2 3 4 --gens 40 --seed $SEED > results_swF2d$SEED.log 2>&1
  $PY scripts/sweep_grades.py --base h11 --mode mechanical --tag swM1d$SEED --grades 2 3 4 --gens 40 --seed $SEED > results_swM1d$SEED.log 2>&1
  echo "seed $SEED done (H2 functional, H1 mechanical)" >> results_final_sweeps_progress.txt
done
echo ALLDONE >> results_final_sweeps_progress.txt
