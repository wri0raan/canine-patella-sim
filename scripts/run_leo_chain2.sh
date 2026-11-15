#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run() {  # tag init gens sigma eval extra-args... (cost-set via COST env)
  tag=$1; init=$2; gens=$3; sig=$4; ev=$5; shift 5
  [ -f results/$tag/done.txt ] || PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$init/best.npz --calibrate --leo --sigma0 $sig --gens $gens --speed 0.85 --gait-set eval_duration=$ev --cost-set $COST "$@" --tag $tag > results_$tag.log 2>&1
}
COST="fall_const=120 w_yawrate=4 w_track=0"                                              ; run leoD1 leoC  200 0.08 8
COST="fall_const=120 w_yawrate=4 w_track=0 w_clear=5 w_spine=2"                          ; run leoD2 leoD1 150 0.05 8
COST="fall_const=120 w_yawrate=4 w_clear=5 w_spine=2 w_track=4"                          ; run leoD3 leoD2 150 0.05 8
COST="fall_const=120 w_yawrate=4 w_clear=5 w_spine=2 w_track=4 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1"; run leoD4 leoD3 250 0.05 14 --w-meta 10
echo CHAIN2_DONE > results_leo_chain2.txt
