#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
BASE="--base leoH3 --leo --speed 0.73 --gens 60 --grades 2 3 4 --w-meta 10 --gait-set eval_duration=14 --cost-set $COST"
run() { tag=$1; shift; [ -f results/${tag}_summary.csv ] && { echo "skip $tag"; return; }; $PY scripts/sweep_grades.py --tag $tag $BASE "$@" >> results_$tag.log 2>&1; echo "$tag done" >> results_leo_sweeps_progress.txt; }
run swLF2s1 --mode functional --style-ref leoH3 --style-w 2.0 --seed 1
run swLM1s1 --mode mechanical --seed 1
run swLF1s1 --mode functional --seed 1
run swLH3s1 --mode functional --style-ref leoH3 --style-w 2.0 --load-avoid 0.10 --seed 1
run swLU2s1 --mode functional --style-ref leoH3 --style-w 2.0 --free-legs LH --affected LH --seed 1
for S in 2 3; do
  run swLF2s$S --mode functional --style-ref leoH3 --style-w 2.0 --seed $S
  run swLM1s$S --mode mechanical --seed $S
  run swLF1s$S --mode functional --seed $S
done
echo ALLDONE >> results_leo_sweeps_progress.txt
