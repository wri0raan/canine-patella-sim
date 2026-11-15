#!/bin/sh
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until grep -q ALLDONE results_final_sweeps_progress.txt 2>/dev/null; do sleep 30; done
for SEED in 2 3; do
  $PY scripts/sweep_grades.py --base h11 --mode functional --tag swF1d$SEED --grades 2 3 4 --gens 40 --seed $SEED > results_swF1d$SEED.log 2>&1
done
echo "design 3x3 complete" >> results_extra_sweeps_progress.txt
$PY scripts/sweep_grades.py --base h11 --mode functional --tag swH3d1 --style-ref h11 --style-w 2.0 --load-avoid 0.10 --grades 2 3 4 --gens 40 --seed 1 > results_swH3d1.log 2>&1
echo "H3 done" >> results_extra_sweeps_progress.txt
$PY scripts/sweep_grades.py --base h11 --mode functional --tag swU2d1 --style-ref h11 --style-w 2.0 --free-legs LH --affected LH --grades 2 3 4 --gens 40 --seed 1 > results_swU2d1.log 2>&1
echo "unilateral done" >> results_extra_sweeps_progress.txt
echo ALLDONE >> results_extra_sweeps_progress.txt
