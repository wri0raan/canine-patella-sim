#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
BASE="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03 stride_target=0 w_freq=3 w_speed=40"
step() {  # tag init grade
  [ -f results/$1/done.txt ] || $PY scripts/optimize_gait.py --grade $3 --mode functional --init results/$2/best.npz --leo --sigma0 0.03 --gens 50 --speed 2.3 \
      --gait-set eval_duration=8 target_speed=2.3 freq_lo=2.80 amp_max=1.2 --cost-set $BASE freq_target=3.10 duty_target=0.40 --w-meta 10 --style-ref trotS4 --style-w 0.3 --spine --tag $1 > results_$1.log 2>&1
  echo "$1 done $(date +%H:%M)" >> results_trot_progress.txt
}
step trotS4_G2 trotS4 2
step trotS4_G3 trotS4_G2 3
echo GRADESDONE >> results_trot_progress.txt
