#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until grep -q ALLDONE results_leo_sweeps_progress.txt 2>/dev/null; do sleep 60; done
COST="fall_const=60 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
[ -f results/swLU2s1b_summary.csv ] || $PY scripts/sweep_grades.py --tag swLU2s1b --base leoH3 --leo --speed 0.73 --gens 100 --sigma0 0.08 --grades 2 3 4 --w-meta 10 --gait-set eval_duration=14 --cost-set $COST --mode functional --style-ref leoH3 --style-w 2.0 --free-legs LH --affected LH --seed 1 >> results_swLU2s1b.log 2>&1
echo RETRY_DONE >> results_leo_sweeps_progress.txt
