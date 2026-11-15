#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run() {  # tag init gens sigma eval [extra args]; COST env = cost-set string
  tag=$1; init=$2; gens=$3; sig=$4; ev=$5; shift 5
  [ -f results/$tag/done.txt ] || PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$init/best.npz --calibrate --leo --sigma0 $sig --gens $gens --speed 0.85 --gait-set eval_duration=$ev --cost-set $COST "$@" --tag $tag > results_$tag.log 2>&1
}
B="fall_const=120 w_yawrate=4 w_track=0"
COST="$B calib_scale=15"                         ; run leoE1 leoC  150 0.08 8
COST="$B calib_scale=10"                         ; run leoE2 leoE1 150 0.06 8
COST="$B calib_scale=6"                          ; run leoE3 leoE2 150 0.05 8
COST="$B calib_scale=6 w_clear=5 w_spine=2"      ; run leoE4 leoE3 150 0.05 8
COST="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=5 w_spine=2" ; run leoE5 leoE4 150 0.05 8
COST="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=5 w_spine=2 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1"; run leoE6 leoE5 250 0.05 14 --w-meta 10
echo CHAIN3_DONE > results_leo_chain3.txt
