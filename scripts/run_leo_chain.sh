#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
T=leoC
BASE="fall_const=120 w_yawrate=4 w_spine=3 w_clear=10"
[ -f results/$T/best.npz ] || $PY scripts/curriculum.py --tag $T --grade 0 --mode functional --speeds 0.15 0.3 0.45 0.6 0.75 0.85 --gens 80 --sigma0 0.12 --extra "--leo --gait-set eval_duration=8 --cost-set fall_const=60 w_yawrate=4 w_spine=3 w_clear=10" > results_${T}.log 2>&1
[ -f results/${T}1/done.txt ] || PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$T/best.npz --calibrate --leo --sigma0 0.08 --gens 200 --speed 0.85 --gait-set eval_duration=8 --cost-set $BASE --tag ${T}1 > results_${T}1.log 2>&1
[ -f results/${T}2/done.txt ] || PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/${T}1/best.npz --calibrate --leo --sigma0 0.05 --gens 250 --speed 0.85 --w-meta 10 --gait-set eval_duration=14 --cost-set $BASE w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 --tag ${T}2 > results_${T}2.log 2>&1
echo CHAIN_DONE > results_leo_chain.txt
