#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
COST="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03 stride_target=0.8"
[ -f results/leoM52/done.txt ] || PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/leoS1/best.npz --calibrate --leo --sigma0 0.04 --gens 80 --speed 0.55 --gait-set eval_duration=14 freq_lo=1.7 target_speed=0.55 --cost-set $COST --w-meta 10 --tag leoM52 > results_leoM52.log 2>&1
echo M52_DONE >> results_leo_sweeps_progress.txt
